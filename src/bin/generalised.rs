use std::{borrow::Cow, fs::OpenOptions, io::Read, time::Instant};

use clustered::{shader_bytes::ShaderBytes, wgpu_map_helper, RunShaderParams};
use rand::{rngs::StdRng, Rng, SeedableRng};
use wgpu::{
    Backends, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, DeviceDescriptor, Features,
    InstanceDescriptor, InstanceFlags, Limits, RequestAdapterOptions, ShaderModuleDescriptor,
};

#[tokio::main]
async fn main() {
    env_logger::init();
    let instance = wgpu::Instance::new(InstanceDescriptor {
        backends: Backends::all(),
        flags: InstanceFlags::empty(),
        dx12_shader_compiler: wgpu::Dx12Compiler::default(),
        gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
    });
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            compatible_surface: None,
            force_fallback_adapter: false,
            power_preference: wgpu::PowerPreference::None,
        })
        .await
        .unwrap();
    println!("Using {:?}", adapter.get_info());
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
        .unwrap();
    let mut cs_source = String::new();
    OpenOptions::new()
        .read(true)
        .write(false)
        .open("shader.wgsl")
        .unwrap()
        .read_to_string(&mut cs_source)
        .unwrap();
    let cs_module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("Compute module"),
        source: wgpu::ShaderSource::Wgsl(Cow::from(cs_source)),
    });

    let mut rng = StdRng::seed_from_u64(2);

    let mut data_total = [0u128; 2];
    let mut data_min = [u128::MAX; 2];
    let mut data_max = [0u128; 2];

    let n_elem = 128 * 1024 * 1024 / 4;
    let n_iter = 100;
    let in_buf = device.create_buffer(&BufferDescriptor {
        label: None,
        size: (n_elem * f32::shader_bytes_size()).try_into().unwrap(),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    use clustered::shader_bytes::ShaderBytesInfo;
    let mut out_buf = device.create_buffer(&BufferDescriptor {
        label: None,
        size: (n_elem * f32::shader_bytes_size()).try_into().unwrap(),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    #[allow(unused_assignments)]
    let mut res2: Vec<f32> = Vec::new();
    use rayon::prelude::*;

    for _ in 0..n_iter {
        let input_data = (0..n_elem)
            .map(|_| rng.gen_range(-std::f32::consts::PI..=std::f32::consts::PI))
            .collect::<Vec<_>>();

        let before_gpu = Instant::now();
        queue.write_buffer(
            &in_buf,
            0,
            ShaderBytes::serialise_from_slice(&input_data).get_data(),
        );

        clustered::run_shader(RunShaderParams {
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
            size: out_buf.size(),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&out_buf, 0, &transfer_buf, 0, out_buf.size());
        queue.submit([encoder.finish()].into_iter());

        let transfer_buf_view = transfer_buf.slice(..);
        wgpu_map_helper(&device, wgpu::MapMode::Read, &transfer_buf_view)
            .await
            .unwrap();
        let res: Vec<f32> =
            ShaderBytes::deserialise_to_slice(&transfer_buf_view.get_mapped_range()).collect();
        let gpu_time = (Instant::now() - before_gpu).as_millis();

        data_total[0] += gpu_time;
        if gpu_time < data_min[0] {
            data_min[0] = gpu_time;
        }
        if gpu_time > data_max[0] {
            data_max[0] = gpu_time;
        }
        // Cleanup resources on the gpu side
        device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        let before_cpu = Instant::now();
        res2 = input_data
            .par_iter()
            .map(|value| {
                let mut e = *value;
                for _ in 0..100 {
                    e = (e * e).sqrt();
                }
                e
            })
            .collect();
        let cpu_time = (Instant::now() - before_cpu).as_millis();
        data_total[1] += cpu_time;
        if cpu_time < data_min[1] {
            data_min[1] = cpu_time;
        }
        if cpu_time > data_max[1] {
            data_max[1] = cpu_time;
        }
        for (i, (e1, e2)) in res.iter().zip(res2.iter()).enumerate() {
            if (e1 - e2).abs() > 0.0001 {
                println!("Mismatch at {}!", i);
                println!("GPU said: {}!", e1);
                println!("CPU said: {}!", e2);
                println!("Input was: {:?}", input_data[i]);
                assert_eq!(e1, e2);
            }
        }
    }

    let avg_cpu = data_total[1] as f64 / n_iter as f64;
    println!(
        "CPU time: {:.2}ms +{:.2} or -{:.2}",
        avg_cpu,
        data_max[1] as f64 - avg_cpu,
        avg_cpu - data_min[1] as f64
    );

    let avg_gpu = data_total[0] as f64 / n_iter as f64;
    println!(
        "GPU time: {:.2}ms +{:.2} or -{:.2}",
        avg_gpu,
        data_max[0] as f64 - avg_gpu,
        avg_gpu - data_min[0] as f64
    );
    println!("GPU is ~{:.2}x faster!", avg_cpu / avg_gpu);
}
