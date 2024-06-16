use std::{borrow::Cow, fs::OpenOptions, io::Read, time::Instant};

use clustered::{shader_bytes::ShaderBytes, RunShaderParams};
use rand::{rngs::StdRng, Rng, SeedableRng};
use wgpu::{DeviceDescriptor, Features, Limits, RequestAdapterOptions, ShaderModuleDescriptor};

#[tokio::main]
async fn main() {
    env_logger::init();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            force_fallback_adapter: true,
            power_preference: wgpu::PowerPreference::LowPower,
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
    to_sort.resize_with(1024 * 64, || rng.gen_range(0u32..=100u32));
    let mut shader_output = vec![0u32; to_sort.len()];

    let mut shader_input = to_sort.as_ref();
    let mut subsize = 1;
    let gpu_before_time = Instant::now();
    loop {
        let shader_complete_input = Info {
            input_a_size: subsize,
            input_b_size: subsize,
            data: shader_input,
        };

        clustered::run_shader(RunShaderParams::<u32> {
            device: &device,
            queue: &queue,
            entry_point: "main",
            in_data: unsafe {
                ShaderBytes::from_raw(&shader_complete_input.to_shader_bytes_custom())
            },
            n_workgroups: usize::div_ceil(
                shader_complete_input.data.len(),
                (shader_complete_input.input_a_size + shader_complete_input.input_b_size)
                    .try_into()
                    .unwrap(),
            ),
            out_data: &mut shader_output,
            program: &cs_module,
            workgroup_len: 1,
        })
        .await;
        shader_input = &shader_output;
        subsize *= 2;
        if subsize >= to_sort.len().try_into().unwrap() {
            break;
        }
    }
    let gpu_time = Instant::now() - gpu_before_time;

    let cpu_before_time = Instant::now();
    to_sort.sort();
    let cpu_time = Instant::now() - cpu_before_time;
    println!("GPU took: {}ms", gpu_time.as_millis());
    println!("CPU took: {}ms", cpu_time.as_millis());
    //println!("{:?}", to_sort);
    //println!("{:?}", shader_output);
    assert!(to_sort == shader_output);
}
