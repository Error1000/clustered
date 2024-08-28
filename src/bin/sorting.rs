use std::{borrow::Cow, fs::OpenOptions, io::Read, time::Instant};

use clustered::{shader_bytes::ShaderBytes, wgpu_map_helper, RunShaderParams};
use rand::{rngs::StdRng, Rng, SeedableRng};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, DeviceDescriptor, Features, Limits,
    RequestAdapterOptions, ShaderModuleDescriptor,
};

#[tokio::main]
async fn main() {
    env_logger::init();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
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
    let mut cs_source = String::new();
    OpenOptions::new()
        .read(true)
        .write(false)
        .open("shader-mergesort.wgsl")
        .unwrap()
        .read_to_string(&mut cs_source)
        .unwrap();
    let cs_module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("Compute module"),
        source: wgpu::ShaderSource::Wgsl(Cow::from(cs_source)),
    });

    #[derive(Clone)]
    struct Info<'a> {
        input_a_size: u32,
        input_b_size: u32,
        data: &'a [u32],
    }

    impl Info<'_> {
        fn to_shader_bytes_custom(&self) -> Vec<u8> {
            let mut res =
                Vec::with_capacity(std::mem::size_of_val(self.data) + core::mem::size_of::<u32>());
            res.extend_from_slice(&self.input_a_size.to_le_bytes());
            res.extend_from_slice(&self.input_b_size.to_le_bytes());
            for e in self.data {
                res.extend_from_slice(&e.to_le_bytes());
            }
            res
        }
    }

    let mut rng = StdRng::seed_from_u64(4);
    let mut to_sort = Vec::new();
    to_sort.resize_with(1024 * 1024 * 16, || rng.gen_range(0u32..=u32::MAX));

    let mut subsize = 1;
    let shader_complete_input = Info {
        input_a_size: subsize,
        input_b_size: subsize,
        data: to_sort.as_ref(),
    };

    let gpu_before_time = Instant::now();
    let mut in_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &shader_complete_input.to_shader_bytes_custom(),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let mut out_buf = device.create_buffer(&BufferDescriptor {
        label: None,
        size: in_buf.size(),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let (mut a, mut b) = (&mut in_buf, &mut out_buf);
    loop {
        clustered::run_shader(RunShaderParams {
            device: &device,
            queue: &queue,
            entry_point: "main",
            in_buf: a,
            out_buf: b,
            n_workgroups: usize::div_ceil(
                shader_complete_input.data.len(),
                (subsize + subsize).try_into().unwrap(),
            ),
            program: &cs_module,
            workgroup_len: 1,
        });
        (a, b) = (b, a);
        subsize *= 2;
        if subsize >= to_sort.len().try_into().unwrap() {
            break;
        }
    }

    let transfer_buf = device.create_buffer(&BufferDescriptor {
        label: None,
        size: a.size(),
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&CommandEncoderDescriptor::default());
    enc.copy_buffer_to_buffer(a, 0, &transfer_buf, 0, a.size());
    queue.submit([enc.finish()].into_iter());

    let transfer_buf_view = transfer_buf.slice((2 * core::mem::size_of::<u32>()) as u64..);
    wgpu_map_helper(&device, wgpu::MapMode::Read, &transfer_buf_view)
        .await
        .unwrap();
    let shader_output: Vec<u32> =
        ShaderBytes::deserialise_to_slice::<u32>(&transfer_buf_view.get_mapped_range()).collect();
    let gpu_time = Instant::now() - gpu_before_time;

    use rayon::prelude::*;
    let cpu_before_time = Instant::now();
    to_sort.par_sort();
    let cpu_time = Instant::now() - cpu_before_time;
    println!("GPU took: {}ms", gpu_time.as_millis());
    println!("CPU took: {}ms", cpu_time.as_millis());
    // println!("{:?}", to_sort);
    // println!("{:?}", shader_output);
    assert!(to_sort == shader_output);
}
