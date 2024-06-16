use image::{codecs::png::PngEncoder, io::Reader as ImageReader, GenericImageView, ImageEncoder};
use std::{borrow::Cow, fs::OpenOptions, io::Read};
use wgpu::{
    util::DeviceExt, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BufferDescriptor,
    BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, DeviceDescriptor, Extent3d,
    Features, ImageDataLayout, InstanceDescriptor, PipelineLayoutDescriptor, ShaderStages,
    TextureDescriptor, TextureUsages, TextureViewDescriptor,
};

#[tokio::main]
async fn main() {
    env_logger::init();
    let instance = wgpu::Instance::new(InstanceDescriptor::default());

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();
    println!("Adapter: {:?}", adapter.get_info());
    let (device, queue) = adapter
        .request_device(
            &DeviceDescriptor {
                label: Some("Required device"),
                required_features: Features::BUFFER_BINDING_ARRAY
                    | Features::STORAGE_RESOURCE_BINDING_ARRAY,
                required_limits: Default::default(),
            },
            None,
        )
        .await
        .unwrap();

    let mut cs_source = String::new();
    OpenOptions::new()
        .read(true)
        .open("shader2.wgsl")
        .unwrap()
        .read_to_string(&mut cs_source)
        .unwrap();

    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Compute module"),
        source: wgpu::ShaderSource::Wgsl(Cow::from(cs_source)),
    });

    let in_img = ImageReader::open("./in.png").unwrap().decode().unwrap();
    // let in_img = in_img.resize(256, 256, image::imageops::FilterType::Lanczos3);
    let raw_img_data: Vec<u8> = in_img.pixels().flat_map(|pixel| pixel.2 .0).collect();
    let input_buf = device.create_texture_with_data(
        &queue,
        &TextureDescriptor {
            label: Some("Input texture"),
            size: Extent3d {
                width: in_img.width(),
                height: in_img.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &raw_img_data,
    );
    drop(in_img);

    let input_buf_view = input_buf.create_view(&TextureViewDescriptor::default());

    let output_buf = device.create_texture(&TextureDescriptor {
        label: Some("Output texture"),
        size: Extent3d {
            width: input_buf.width(),
            height: input_buf.height(),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let output_buf_view = output_buf.create_view(&TextureViewDescriptor::default());

    let bind_group_0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                count: None,
                visibility: ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
            },
            BindGroupLayoutEntry {
                binding: 1,
                count: None,
                visibility: ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_0_layout],
        label: None,
        push_constant_ranges: &[],
    });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        entry_point: "main",
        label: None,
        module: &cs_module,
        layout: Some(&pipeline_layout),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_0_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&input_buf_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&output_buf_view),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.insert_debug_marker("Compute stuff");
        cpass.dispatch_workgroups(input_buf.width(), input_buf.height(), 1);
    }

    let transfer_buf = device.create_buffer(&BufferDescriptor {
        label: Some("Compute transfer buffer"),
        size: (raw_img_data.len()) as u64,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTextureBase {
            texture: &output_buf,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBufferBase {
            buffer: &transfer_buf,
            layout: ImageDataLayout {
                offset: 0,
                bytes_per_row: Some((core::mem::size_of::<[u8; 4]>() as u32) * output_buf.width()),
                rows_per_image: Some(output_buf.height()),
            },
        },
        output_buf.size(),
    );
    queue.submit(Some(encoder.finish()));

    let output_buf_slice = transfer_buf.slice(..);
    let (ch_send, ch_recv) = flume::bounded(1);
    output_buf_slice.map_async(wgpu::MapMode::Read, move |rez| {
        ch_send.send(rez).unwrap();
    });
    device.poll(wgpu::Maintain::wait()).panic_on_timeout();
    if let Ok(Ok(())) = ch_recv.recv_async().await {
        let data = output_buf_slice.get_mapped_range();
        let result: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
        PngEncoder::new(
            OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open("out.png")
                .unwrap(),
        )
        .write_image(
            &result,
            output_buf.width(),
            output_buf.height(),
            image::ExtendedColorType::Rgba8,
        )
        .unwrap();
        println!("Yay!");
    } else {
        println!("Didn't recive result!");
    }
}
