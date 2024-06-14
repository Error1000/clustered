use shader_bytes::{FromShaderBytes, IntoShaderBytes};
use std::num::NonZeroUsize;
use tokio::task::yield_now;
use wgpu::{
    util::DeviceExt, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BufferDescriptor, BufferSlice, BufferUsages, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipelineDescriptor, Device, PipelineLayoutDescriptor, Queue,
    ShaderModule, ShaderStages,
};

pub mod shader_bytes;

async fn wgpu_map_helper(
    device: &wgpu::Device,
    mode: wgpu::MapMode,
    buf_view: &BufferSlice<'_>,
) -> Result<(), wgpu::BufferAsyncError> {
    let (sender, reciver) = flume::bounded(1);
    buf_view.map_async(mode, move |mapping_res| sender.send(mapping_res).unwrap());
    loop {
        device.poll(wgpu::MaintainBase::Poll);
        yield_now().await;
        if !reciver.is_empty() {
            break;
        }
    }
    reciver
        .recv_async()
        .await
        .expect("Channel should not error out when mapping!")
}

fn div_round_up(a: usize, b: usize) -> usize {
    let res = (a + b - 1) / b;
    if a % b == 0 {
        assert_eq!(res, a / b);
    } else {
        assert_eq!(res, (a / b) + 1);
    }
    res
}

pub struct RunShaderParams<'a, T, U> {
    pub device: &'a Device,
    pub queue: &'a Queue,
    pub in_data: &'a [T],
    pub out_data: &'a mut [U],
    pub workgroup_len: usize,
    pub n_workgroups: Option<NonZeroUsize>,
    pub program: &'a ShaderModule,
    pub entry_point: &'a str,
}
/* IDEA: This could maybe benefit from interning literally everything but the data
   NOTE: Uses output slice length and input slice length are important and used to calculate the size of the shader input and output buffers
   NOTE: Assumes bind group 0 is used for the input and output
   NOTE: Assumes that the same buffer can't be used for input and output
         ^ These are not design choices, these can be changed if wanted
   NOTE: The whole architecture assumes all Ts are of the same size when serialized and all Us are of the same size when serialized
         ^ That one is a design choice, though it wouldn't be a bad idea to allow the serialization some kind of extra buffer to use to serialize that is also uploaded, like some kind of context
   WARNING: Because the input data is serialized for the shader to be able to read,
            type erasure effectively takes place, meaning unless you programmed the shader
            to read the data correctly it won't know what type the data is
            and can easily lead to accidental type punning
   WARNING: If the number of elements in the input is not a multiple of the workgroup size
            then this function will call the shader with global ids outside of the input buffer
            because it is designed to never leave an element unprocessed ( if n_workgroups is None it will auto calculate enough workgroups such that there will be at least one shader invocation per element in the input )
            so if you have 3 elements but a workgroup size of 256, well it still has to do *one* dispatch ( by default ) of a workgroup
            which is 256 in size, so 256-3=253 of those workers in the workgroup will get a global
            id that is out of bounds.
   NOTE:    This function won't try to pad out your buffer for you, this is because *you* can do that yourself
            by giving it an in_data with a len that is a multiple of workgroup_len.
            And on top of that by not doing its own thing, it also allows for other solutions
            like bounds checking in the shader.
            So by not dealing with it it allows more flexibility to the user of this api,
            in other words, it's a feature not a bug.
   NOTE: If n_workgroups is None it will auto calculate enough workgroups such that there will be at least one shader invocation per element in the input
*/

// TODO: In the future when optimizing if the serialization of the input is taking too long, consider changing the api so that the serailization is not necessary
// TODO: Experiment with Features::MAPPABLE_PRIMARY_BUFFERS for extra performance
pub async fn run_shader<T, U>(params: RunShaderParams<'_, T, U>) -> Option<()>
where
    T: Clone + IntoShaderBytes,
    U: FromShaderBytes,
{
    assert!(!params.out_data.is_empty());
    if params.workgroup_len == 0 {
        println!("Your workgroups must have a size of at least 1.");
        return None;
    }

    let n_workgroups: usize = match params.n_workgroups {
        None => div_round_up(params.in_data.len(), params.workgroup_len),
        Some(value) => value.into(),
    };

    let mut serialised_input = vec![0; T::shader_bytes_size() * params.in_data.len()];
    for (i, chnk) in serialised_input
        .chunks_mut(T::shader_bytes_size())
        .enumerate()
    {
        params.in_data[i].to_shader_bytes(chnk);
    }

    let in_buf = params
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input compute data buffer"),
            contents: &serialised_input,
            usage: BufferUsages::STORAGE,
        });

    let out_buf = params.device.create_buffer(&BufferDescriptor {
        label: Some("Output compute data buffer"),
        size: (params.out_data.len() * U::shader_bytes_size())
            .try_into()
            .unwrap(),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut metadata_var = [0u8; core::mem::size_of::<u32>()];
    let meta_buf = params.device.create_buffer(&BufferDescriptor {
        size: metadata_var.len() as u64,
        mapped_at_creation: false,
        label: Some("Metadata compute uniform buffer"),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
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
                        min_binding_size: Some(in_buf.size().try_into().unwrap()),
                    },
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    count: None,
                    visibility: ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(out_buf.size().try_into().unwrap()),
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
        });

    let bind_group_0 = params.device.create_bind_group(&BindGroupDescriptor {
        label: Some("Bind group 0"),
        layout: &bind_group_0_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: in_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: meta_buf.as_entire_binding(),
            },
        ],
    });

    let transfer_buf = params.device.create_buffer(&BufferDescriptor {
        label: Some("Intermediary buffer"),
        size: out_buf.size(),
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
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

    let remaining_workgroups = n_workgroups % max_dispatch_workgroups;
    // We try to dispatch as many workgroups per pass as possible and deal with the remainder afterwards
    for workgroup_id in (0..n_workgroups - remaining_workgroups).step_by(max_dispatch_workgroups) {
        // Tell the compute shader its absolute offset ( in elements )
        // because the global offset is only global within the dispatch
        u32::to_shader_bytes(
            &u32::try_from(workgroup_id * params.workgroup_len).unwrap(),
            &mut metadata_var,
        );
        params.queue.write_buffer(&meta_buf, 0, &metadata_var);
        dispatch_workgroups(u32::try_from(max_dispatch_workgroups).unwrap());
    }

    // Deal with remainder
    if remaining_workgroups != 0 {
        u32::to_shader_bytes(
            &u32::try_from((n_workgroups - remaining_workgroups) * params.workgroup_len).unwrap(),
            &mut metadata_var,
        );
        params.queue.write_buffer(&meta_buf, 0, &metadata_var);
        dispatch_workgroups(u32::try_from(remaining_workgroups).unwrap());
    }

    // Transfer data
    let mut final_encoder = params
        .device
        .create_command_encoder(&CommandEncoderDescriptor { label: None });
    final_encoder.copy_buffer_to_buffer(&out_buf, 0, &transfer_buf, 0, out_buf.size());
    params.queue.submit(Some(final_encoder.finish()));

    let transfer_buf_view = transfer_buf.slice(..);
    if let Ok(()) = wgpu_map_helper(params.device, wgpu::MapMode::Read, &transfer_buf_view).await {
        for (i, val) in transfer_buf_view
            .get_mapped_range()
            .chunks_exact(U::shader_bytes_size())
            .map(|raw_bytes| U::from_shader_bytes(raw_bytes))
            .enumerate()
        {
            params.out_data[i] = val;
        }
        return Some(());
    }

    None
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use rand::{rngs::StdRng, Rng, SeedableRng};
    use wgpu::{
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

        let mut res = vec![0u32; n_elem];
        run_shader::<u32, u32>(RunShaderParams {
            device: &device,
            queue: &queue,
            in_data: &input_data,
            out_data: &mut res,
            workgroup_len: 32,
            n_workgroups: None,
            program: &cs_module,
            entry_point: "main",
        })
        .await
        .unwrap();

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
