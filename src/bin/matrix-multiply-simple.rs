#[path = "../bin-utils/matrix.rs"]
mod matrix;
use matrix::*;

use std::{borrow::Cow, fs::OpenOptions, io::Read, time::Instant};

use clustered::{shader_bytes::ShaderBytes, wgpu_map_helper, RunShaderParams};
use rand::{rngs::StdRng, Rng, SeedableRng};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, DeviceDescriptor, Features,
    InstanceDescriptor, RequestAdapterOptions, ShaderModuleDescriptor,
};

struct InData<'a> {
    matrix1_ncols: u32,
    matrix1_nrows: u32,
    matrix2_ncols: u32,
    // matrix2_nrows == matrix1_ncols
    output_matrix_order: u32, // 1 = column major, 2 = row major
    in_matrix_data: Cow<'a, [f32]>,
}

impl<'a> InData<'a> {
    // NOTE: Allocates a new area to copy the two matrices into one contiguous memory area which can be used for the shader buffer
    fn from(
        left: &RowMajorMatrix<f32>,
        right: &ColMajorMatrix<f32>,
        output_matrix_order: u32,
    ) -> InData<'a> {
        assert!(left.ncols == right.nrows);
        assert!(output_matrix_order == 1 || output_matrix_order == 2);
        let mut formatted_data =
            Vec::<f32>::with_capacity(left.get_n_elems() + right.get_n_elems());
        formatted_data.extend(left.data.iter());
        formatted_data.extend(right.data.iter());
        InData {
            matrix1_ncols: left.ncols,
            matrix1_nrows: left.nrows,
            matrix2_ncols: right.ncols,
            // matrix2_nrows == matrix1_ncols,
            output_matrix_order,
            in_matrix_data: Cow::from(formatted_data),
        }
    }

    fn into_shader_bytes(self) -> Vec<u8> {
        let mut res = Vec::<u8>::new();
        res.extend(self.matrix1_ncols.to_le_bytes());
        res.extend(self.matrix1_nrows.to_le_bytes());
        res.extend(self.matrix2_ncols.to_le_bytes());
        res.extend(self.output_matrix_order.to_le_bytes());
        res.extend(
            self.in_matrix_data
                .iter()
                .flat_map(|val| val.to_le_bytes().into_iter()),
        );
        res
    }
}

#[tokio::main]
async fn main() {
    let instance = wgpu::Instance::new(InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            ..Default::default()
        })
        .await
        .unwrap();
    println!("Using: {:?}", adapter.get_info());
    let (device, queue) = adapter
        .request_device(
            &DeviceDescriptor {
                required_features: Features::STORAGE_RESOURCE_BINDING_ARRAY
                    | Features::BUFFER_BINDING_ARRAY,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
    let mut cs_source = String::new();
    OpenOptions::new()
        .read(true)
        .write(false)
        .open("shader-matrix-mult-chunked.wgsl") // Note: can also use shader-matrix-mult-simple
        .unwrap()
        .read_to_string(&mut cs_source)
        .unwrap();
    let cs_module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("Compute module"),
        source: wgpu::ShaderSource::Wgsl(Cow::from(cs_source)),
    });

    let mut buf = String::new();
    std::io::stdin().read_line(&mut buf).unwrap();
    let mut rng = StdRng::seed_from_u64(buf.trim().parse::<u64>().unwrap());
    drop(buf);
    //let mut rng = StdRng::from_entropy();
    let mut left_mat = RowMajorMatrix::new(4000, 4000);
    let mut right_mat = ColMajorMatrix::new(4000, 4000);

    for i in 0..left_mat.nrows() {
        for j in 0..left_mat.ncols() {
            left_mat[(i, j)] = rng.gen();
        }
    }

    for i in 0..right_mat.nrows() {
        for j in 0..right_mat.ncols() {
            right_mat[(i, j)] = rng.gen();
        }
    }

    let out_matrix_type = 2;
    let out_mat_nrows = left_mat.nrows;
    let out_mat_ncols = right_mat.ncols;
    println!(
        "Output will be {} cols x {} rows!",
        out_mat_ncols, out_mat_nrows
    );
    let time_start = Instant::now();
    assert!(left_mat.ncols == right_mat.nrows);
    let in_data = InData::from(&left_mat, &right_mat, out_matrix_type);

    let in_buf = device.create_buffer_init(&BufferInitDescriptor {
        contents: &in_data.into_shader_bytes(),
        label: None,
        usage: BufferUsages::STORAGE,
    });

    let mut out_buf = device.create_buffer(&BufferDescriptor {
        label: None,
        size: u64::try_from(
            core::mem::size_of::<f32>() * usize::try_from(out_mat_ncols * out_mat_nrows).unwrap(),
        )
        .unwrap(),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    clustered::run_shader(RunShaderParams {
        device: &device,
        queue: &queue,
        program: &cs_module,
        entry_point: "main",
        in_buf: &in_buf,
        out_buf: &mut out_buf,
        n_workgroups: usize::div_ceil(usize::try_from(out_mat_ncols * out_mat_nrows).unwrap(), 32)
            * 32, /* 32 chunks per element */
        workgroup_len: 32,
    });

    let transfer_buf = device.create_buffer(&BufferDescriptor {
        label: None,
        size: out_buf.size(),
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    assert!(out_buf.size() == transfer_buf.size());

    let mut enc = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(&out_buf, 0, &transfer_buf, 0, out_buf.size());
    queue.submit([enc.finish()].into_iter());

    let transfer_view = transfer_buf.slice(..);
    wgpu_map_helper(&device, wgpu::MapMode::Read, &transfer_view)
        .await
        .unwrap();

    assert!(out_matrix_type == 2);
    let res = RowMajorMatrix {
        nrows: out_mat_nrows,
        ncols: out_mat_ncols,
        data: ShaderBytes::deserialise_to_iterator(&transfer_view.get_mapped_range())
            .collect::<Vec<f32>>(),
    };
    let time_end = Instant::now();
    assert!(res.data.len() == usize::try_from(out_mat_nrows * out_mat_ncols).unwrap());
    // println!("{:?}", res);
    println!("Took {}s!", (time_end - time_start).as_secs_f64());
}
