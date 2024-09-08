use std::{borrow::Cow, time::Instant};

use clustered::{shader_bytes::ShaderBytes, wgpu_map_helper, RunShaderParams};
use futures::future::join_all;
use rand::{rngs::StdRng, Rng, SeedableRng};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, DeviceDescriptor, Features,
    InstanceDescriptor, Limits, RequestAdapterOptions, ShaderModuleDescriptor,
};

#[tokio::main]
async fn main() {
    env_logger::init();
    const SHDR: &str = r#"
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
        var res: u32 = 1;
        for(var i = 0; i < 1000; i++){
            res *= e;
        }
        v_out_data[actual_id] = res;
    }
    "#;
    let instance = wgpu::Instance::new(InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            force_fallback_adapter: false,
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .unwrap();
    println!("Using {:?}", adapter.get_info());
    let (device, queue) = adapter
        .request_device(
            &DeviceDescriptor {
                label: None,
                required_features: Features::STORAGE_RESOURCE_BINDING_ARRAY
                    | Features::BUFFER_BINDING_ARRAY,
                required_limits: Limits::default(),
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
    let sh_module = device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::from(SHDR)),
    });

    let n_elements = 128 * 1024;
    let mut futures: Vec<_> = Vec::new();

    for _ in 0..100 {
        let fut = async {
            let mut rng = StdRng::seed_from_u64(4);
            let n_elem = 128 * 1024;
            let mut inv = Vec::<u32>::new();
            inv.resize_with(n_elem, || rng.gen_range(0..=1000));

            let in_buf = device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: ShaderBytes::serialise_from_slice(&inv).get_data(),
                usage: BufferUsages::STORAGE,
            });

            let mut out_buf = device.create_buffer(&BufferDescriptor {
                label: None,
                size: (n_elements * core::mem::size_of::<u32>())
                    .try_into()
                    .unwrap(),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            clustered::run_shader(RunShaderParams {
                device: &device,
                queue: &queue,
                in_buf: &in_buf,
                out_buf: &mut out_buf,
                workgroup_len: 32,
                n_workgroups: usize::div_ceil(inv.len(), 32),
                program: &sh_module,
                entry_point: "main",
            })
            .unwrap();
            let transfer_buf = device.create_buffer(&BufferDescriptor {
                label: None,
                size: out_buf.size(),
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let mut enc = device.create_command_encoder(&CommandEncoderDescriptor::default());
            enc.copy_buffer_to_buffer(&out_buf, 0, &transfer_buf, 0, out_buf.size());
            queue.submit([enc.finish()].into_iter());

            let transfer_buf_view = transfer_buf.slice(..);
            wgpu_map_helper(&device, wgpu::MapMode::Read, &transfer_buf_view)
                .await
                .unwrap();
            let x = ShaderBytes::deserialise_to_iterator(&transfer_buf_view.get_mapped_range())
                .collect::<Vec<u32>>();
            x
        };
        futures.push(fut);
    }

    let time_before_par = Instant::now();
    let par_result = join_all(futures).await;
    let time_par = (Instant::now() - time_before_par).as_millis();
    println!("Parallel run_shader: {:?}ms", time_par);

    let mut seq_result = Vec::<Vec<u32>>::new();
    let time_before_seq = Instant::now();
    for _ in 0..100 {
        let fut = async {
            let mut rng = StdRng::seed_from_u64(4);
            let n_elem = 128 * 1024;
            let mut inv = Vec::<u32>::new();
            inv.resize_with(n_elem, || rng.gen_range(0..=1000));
            let in_buf = device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: ShaderBytes::serialise_from_slice(&inv).get_data(),
                usage: BufferUsages::STORAGE,
            });

            let mut out_buf = device.create_buffer(&BufferDescriptor {
                label: None,
                size: (n_elements * core::mem::size_of::<u32>())
                    .try_into()
                    .unwrap(),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            clustered::run_shader(RunShaderParams {
                device: &device,
                queue: &queue,
                in_buf: &in_buf,
                out_buf: &mut out_buf,
                workgroup_len: 32,
                n_workgroups: usize::div_ceil(inv.len(), 32),
                program: &sh_module,
                entry_point: "main",
            })
            .unwrap();
            let transfer_buf = device.create_buffer(&BufferDescriptor {
                label: None,
                size: out_buf.size(),
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let mut enc = device.create_command_encoder(&CommandEncoderDescriptor::default());
            enc.copy_buffer_to_buffer(&out_buf, 0, &transfer_buf, 0, out_buf.size());
            queue.submit([enc.finish()].into_iter());

            let transfer_buf_view = transfer_buf.slice(..);
            wgpu_map_helper(&device, wgpu::MapMode::Read, &transfer_buf_view)
                .await
                .unwrap();
            let x = ShaderBytes::deserialise_to_iterator(&transfer_buf_view.get_mapped_range())
                .collect::<Vec<u32>>();
            x
        };
        seq_result.push(fut.await);
    }

    let time_seq = (Instant::now() - time_before_seq).as_millis();
    println!("Sequential run_shader: {:?}ms", time_seq);
    assert_eq!(seq_result, par_result);
}
