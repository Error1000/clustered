use shader_bytes::IntoShaderBytes;
use tokio::task::yield_now;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BufferDescriptor, BufferSlice, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipelineDescriptor, Device, PipelineLayoutDescriptor, Queue, ShaderModule, ShaderStages,
};

pub mod networking;
pub mod serialisable_program;
pub mod shader_bytes;

// NOTE: Device is used only for polling
pub async fn wgpu_map_helper(
    device: &wgpu::Device,
    mode: wgpu::MapMode,
    buf_view: &BufferSlice<'_>,
) -> Result<(), wgpu::BufferAsyncError> {
    let (sender, receiver) = flume::bounded(1);
    buf_view.map_async(mode, move |mapping_res| {
        tokio::spawn(async move {
            if let Err(err) = mapping_res.clone() {
                println!("Error: Mapping failed with error: {err}!");
            }

            if let Err(err) = sender.try_send(mapping_res) {
                panic!(
                    "Error: Failed to send mapping result over flume channel, error was: {err}!"
                );
            }
        });
    });

    loop {
        device.poll(wgpu::MaintainBase::Poll).panic_on_timeout();
        yield_now().await;
        if !receiver.is_empty() {
            break;
        }
    }
    receiver
        .recv_async()
        .await
        .expect("Channel should not error out when receiving mapping result!")
}

pub struct RunShaderParams<'a> {
    pub device: &'a Device,
    pub queue: &'a Queue,
    pub in_buf: &'a wgpu::Buffer,
    pub out_buf: &'a mut wgpu::Buffer,
    pub workgroup_len: usize,
    pub n_workgroups: usize,
    pub program: &'a ShaderModule,
    pub entry_point: &'a str,
}

/* IDEA: This could maybe benefit from interning literally everything but the data
   NOTE: Assumes bind group 0 is used for the input and output
   NOTE: Assumes that the same buffer can't be used for input and output
         ^ These are not design choices, these can be changed if wanted
   WARNING: Because the input data is serialized for the shader to be able to read,
            type erasure effectively takes place, meaning unless you programmed the shader
            to read the data correctly it won't know what type the data is
            and can easily lead to accidental type punning
   WARNING: This function will call the shader with global ids up to workgroup_len*n_workgroups, this means
            it can and *will* call the shader with global ids outside the *length* of the input buffer if told to do so.
   NOTE:    This function won't try to pad out your buffer for you, this is because *you* can do that yourself.
   NOTE:    Total number of calls = number of workgroups * workgroup len
*/

// TODO: Experiment with Features::MAPPABLE_PRIMARY_BUFFERS for extra performance

pub fn run_shader(params: RunShaderParams<'_>) -> Option<()> {
    assert!(params.out_buf.size() != 0);
    assert!(params.in_buf.size() != 0);
    if params.workgroup_len == 0 {
        println!("Your workgroups must have a size of at least 1.");
        return None;
    }
    let n_workgroups: usize = params.n_workgroups;
    assert!(n_workgroups != 0);

    let mut metadata_var = [0u8; core::mem::size_of::<u32>()];
    let meta_buf = params.device.create_buffer(&BufferDescriptor {
        label: Some("Metadata compute uniform buffer"),
        size: metadata_var.len() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group_0_layout = params
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Compute pipeline bind group layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    count: None,
                    visibility: ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(params.in_buf.size().try_into().unwrap()),
                    },
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    count: None,
                    visibility: ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(params.out_buf.size().try_into().unwrap()),
                    },
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    count: None,
                    visibility: ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(meta_buf.size().try_into().unwrap()),
                    },
                },
            ],
        });

    let compute_pipeline_layout = params
        .device
        .create_pipeline_layout(&PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_0_layout],
            label: Some("Compute pipeline layout"),
            push_constant_ranges: &[],
        });

    let compute_pipeline = params
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            entry_point: params.entry_point,
            label: Some("Compute pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: params.program,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let bind_group_0 = params.device.create_bind_group(&BindGroupDescriptor {
        label: Some("Bind group 0"),
        layout: &bind_group_0_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: params.in_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: params.out_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: meta_buf.as_entire_binding(),
            },
        ],
    });

    let dispatch_workgroups = |how_many| {
        let mut encoder = params
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group_0, &[]);
            cpass.dispatch_workgroups(how_many, 1, 1);
        }

        params.queue.submit(Some(encoder.finish()));
    };

    let max_dispatch_workgroups: usize = params
        .device
        .limits()
        .max_compute_workgroups_per_dimension
        .try_into()
        .unwrap();

    let remainder_workgroups = n_workgroups % max_dispatch_workgroups;

    // We try to dispatch as many workgroups per pass as possible and deal with the remainder afterwards
    for workgroup_id in (0..n_workgroups - remainder_workgroups).step_by(max_dispatch_workgroups) {
        // Tell the compute shader its absolute offset
        // because the global offset is only global within the dispatch
        u32::to_shader_bytes(
            &u32::try_from(workgroup_id * params.workgroup_len).unwrap(),
            &mut metadata_var,
        );
        params.queue.write_buffer(&meta_buf, 0, &metadata_var);
        dispatch_workgroups(u32::try_from(max_dispatch_workgroups).unwrap());
    }

    // Deal with remainder
    if remainder_workgroups != 0 {
        u32::to_shader_bytes(
            &u32::try_from((n_workgroups - remainder_workgroups) * params.workgroup_len).unwrap(),
            &mut metadata_var,
        );
        params.queue.write_buffer(&meta_buf, 0, &metadata_var);
        dispatch_workgroups(u32::try_from(remainder_workgroups).unwrap());
    }

    Some(())
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use rand::{rngs::StdRng, Rng, SeedableRng};
    use shader_bytes::ShaderBytes;
    use wgpu::{
        util::{BufferInitDescriptor, DeviceExt},
        DeviceDescriptor, Features, InstanceDescriptor, Limits, RequestAdapterOptions,
        ShaderModuleDescriptor,
    };

    use super::*;

    #[tokio::test]
    async fn test_computation_equivalence() {
        let instance = wgpu::Instance::new(InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                force_fallback_adapter: false,
                power_preference: wgpu::PowerPreference::None,
                ..Default::default()
            })
            .await
            .expect("Adapter must exist!");
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: None,
                    required_features: Features::BUFFER_BINDING_ARRAY
                        | Features::STORAGE_RESOURCE_BINDING_ARRAY,
                    required_limits: Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .expect("Device must have required features!");
        const CS_SOURCE: &str = r#"
                @group(0)
                @binding(0)
                var<storage, read> v_in_data: array<u32>;

                @group(0)
                @binding(1)
                var<storage, read_write> v_out_data: array<u32>;

                @group(0)
                @binding(2)
                var<uniform> goff: u32;

                @compute
                @workgroup_size(32)
                fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                    let actual_id = gid.x+goff;
                    if (actual_id >= arrayLength(&v_in_data)){ return; }
                    if (actual_id >= arrayLength(&v_out_data)){ return; }
                    var e = v_in_data[actual_id];
                    v_out_data[actual_id] = e*e;
                }
            "#;
        let cs_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Compute module"),
            source: wgpu::ShaderSource::Wgsl(Cow::from(CS_SOURCE)),
        });

        let mut rng = StdRng::seed_from_u64(2);

        let n_elem = 1024 * 1024;

        let input_data = (0..n_elem)
            .map(|_| rng.gen_range(0u32..=1000u32))
            .collect::<Vec<_>>();

        let mut out_buf = device.create_buffer(&BufferDescriptor {
            label: None,
            size: (n_elem * core::mem::size_of::<u32>()).try_into().unwrap(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let in_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: &ShaderBytes::serialise_from_slice(&input_data).into_data(),
            usage: BufferUsages::STORAGE,
        });

        run_shader::<u32>(RunShaderParams {
            device: &device,
            queue: &queue,
            in_buf: &in_buf,
            out_buf: &mut out_buf,
            workgroup_len: 32,
            n_workgroups: usize::div_ceil(input_data.len(), 32),
            program: &cs_module,
            entry_point: "main",
        })
        .await
        .unwrap();

        let transfer_buf = device.create_buffer(&BufferDescriptor {
            label: None,
            mapped_at_creation: false,
            size: out_buf.size(),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&out_buf, 0, &transfer_buf, 0, out_buf.size());
        queue.submit([encoder.finish()].into_iter());

        let transfer_buf_view = transfer_buf.slice(..);
        wgpu_map_helper(&device, wgpu::MapMode::Read, &transfer_buf_view)
            .await
            .unwrap();
        let res: Vec<u32> =
            ShaderBytes::deserialise_to_slice(&transfer_buf_view.get_mapped_range());
        drop(transfer_buf);

        // Cleanup resources on the gpu side
        device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        use rayon::prelude::*;
        let res2: Vec<u32> = input_data.par_iter().map(|value| value * value).collect();

        assert_eq!(res.len(), res2.len());
        for (i, (e1, e2)) in res.iter().zip(res2.iter()).enumerate() {
            if e1 != e2 {
                println!("Mismatch at {}!", i);
                println!("GPU said: {}!", e1);
                println!("CPU said: {}!", e2);
                println!("Input was: {:?}", input_data[i]);
                assert_eq!(e1, e2);
            }
        }
    }
}
