use std::{borrow::Cow, time::Instant};

use clustered::RunShaderParams;
use futures::future::join_all;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::join;
use tokio::join;
use wgpu::{
    DeviceDescriptor, Features, InstanceDescriptor, InstanceFlags, Limits, RequestAdapterOptions,
    ShaderModuleDescriptor,
};

#[tokio::main]
async fn main() {
    const sh: &str = r#"
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
        v_out_data[actual_id] = e+1;
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
            },
            None,
        )
        .await
        .unwrap();
    let sh_module = device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::from(sh)),
    });

    let mut futures: Vec<_> = Vec::new();

    for _ in 0..100 {
        let fut = async {
            let mut rng = StdRng::seed_from_u64(4);
            let mut outv = vec![0; 256 * 128 * 128];
            let mut inv = Vec::new();
            inv.resize_with(256 * 128 * 128, || rng.gen_range(0..=1000));
            clustered::run_shader::<u32, u32>(RunShaderParams {
                device: &device,
                queue: &queue,
                in_data: &inv,
                out_data: &mut outv,
                workgroup_len: 32,
                n_workgroups: None,
                program: &sh_module,
                entry_point: "main",
            })
            .await
            .unwrap();
        };
        futures.push(fut);
    }

    let time_before_par = Instant::now();
    join_all(futures).await;
    let time_par = (Instant::now() - time_before_par).as_millis();
    println!("Parallel run_shader: {:?}ms", time_par);

    let time_before_seq = Instant::now();
    for _ in 0..100 {
        let fut = async {
            let mut rng = StdRng::seed_from_u64(4);
            let mut outv = vec![0; 256 * 128 * 128];
            let mut inv = Vec::new();
            inv.resize_with(256 * 128 * 128, || rng.gen_range(0..=1000));
            clustered::run_shader::<u32, u32>(RunShaderParams {
                device: &device,
                queue: &queue,
                in_data: &inv,
                out_data: &mut outv,
                workgroup_len: 32,
                n_workgroups: None,
                program: &sh_module,
                entry_point: "main",
            })
            .await
            .unwrap();
        };
        fut.await;
    }

    let time_seq = (Instant::now() - time_before_seq).as_millis();
    println!("Sequential run_shader: {:?}ms", time_seq);
}
