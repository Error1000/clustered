use std::{
    borrow::Cow,
    fmt::Debug,
    fs::OpenOptions,
    io::Read,
    ops::{Index, IndexMut},
    time::Instant,
};

use clustered::{shader_bytes::ShaderBytes, wgpu_map_helper, RunShaderParams};
use rand::{rngs::StdRng, Rng, SeedableRng};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, DeviceDescriptor, Features,
    InstanceDescriptor, RequestAdapterOptions, ShaderModuleDescriptor,
};

macro_rules! matrix_impl {
    ($struct_name:ident) => {
        impl Index<(usize, usize)> for $struct_name {
            type Output = f32;
            fn index(&self, index: (usize, usize)) -> &Self::Output {
                let off = self.index_to_offset(index);
                &self.data[off]
            }
        }

        impl IndexMut<(usize, usize)> for $struct_name {
            fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
                let off = self.index_to_offset(index);
                &mut self.data[off]
            }
        }

        impl Debug for $struct_name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                for i in 0..self.nrows {
                    for j in 0..self.ncols {
                        write!(
                            f,
                            "{:?} ",
                            self[(i.try_into().unwrap(), j.try_into().unwrap())]
                        )?;
                    }
                    writeln!(f)?;
                }
                Ok(())
            }
        }
    };
}

#[derive(Clone, PartialEq)]
struct ColMajorMatrix {
    ncols: u32,
    nrows: u32,
    data: Vec<f32>,
}

impl ColMajorMatrix {
    fn new(nrows: u32, ncols: u32) -> Self {
        let mut inner = Vec::<f32>::new();
        inner.resize(usize::try_from(ncols * nrows).unwrap(), 0u8.into());
        Self {
            ncols,
            nrows,
            data: inner,
        }
    }

    fn index_to_offset(&self, index: (usize, usize)) -> usize {
        // i + j * number_of_elems_in_column
        index.0 + index.1 * (usize::try_from(self.nrows).unwrap())
    }

    fn get_n_elems(&self) -> u32 {
        self.nrows * self.ncols
    }

    fn transpose_lazy(self) -> RowMajorMatrix {
        RowMajorMatrix {
            nrows: self.ncols,
            ncols: self.nrows,
            data: self.data,
        }
    }
}

matrix_impl!(ColMajorMatrix);

#[derive(Clone, PartialEq)]
struct RowMajorMatrix {
    ncols: u32,
    nrows: u32,
    data: Vec<f32>,
}

impl RowMajorMatrix {
    fn new(nrows: u32, ncols: u32) -> Self {
        let mut inner = Vec::<f32>::new();
        inner.resize(usize::try_from(ncols * nrows).unwrap(), 0u8.into());
        Self {
            ncols,
            nrows,
            data: inner,
        }
    }

    fn index_to_offset(&self, index: (usize, usize)) -> usize {
        // j + i * number_of_elems_in_row
        index.1 + index.0 * (usize::try_from(self.ncols).unwrap())
    }

    fn get_n_elems(&self) -> u32 {
        self.nrows * self.ncols
    }

    fn mult(&self, other: &ColMajorMatrix) -> RowMajorMatrix {
        assert!(self.ncols == other.nrows);
        use rayon::prelude::*;
        RowMajorMatrix {
            nrows: self.nrows,
            ncols: other.ncols,
            data: (0..self.nrows)
                .into_par_iter()
                .flat_map(|i| {
                    (0..other.ncols).into_par_iter().map(move |j| {
                        // The calculation of one element is not parallelized
                        (0..self.ncols)
                            .map(move |k| {
                                self[(i.try_into().unwrap(), k.try_into().unwrap())]
                                    * other[(k.try_into().unwrap(), j.try_into().unwrap())]
                            })
                            .sum()
                    })
                })
                .collect(),
        }
    }

    fn transpose_lazy(self) -> ColMajorMatrix {
        ColMajorMatrix {
            ncols: self.nrows,
            nrows: self.ncols,
            data: self.data,
        }
    }
}

matrix_impl!(RowMajorMatrix);

struct InData<'a> {
    matrix1_ncols: u32,
    matrix1_nrows: u32,
    matrix2_ncols: u32,
    // matrix2_nrows == matrix1_ncols
    output_matrix_order: u32, // 1 = column major, 2 = row major
    in_matrix_data: Cow<'a, [f32]>,
}

impl<'a> InData<'a> {
    // NOTE: Allocates a new area to copy the two matrices into one contigous memory area which can be used for the shader buffer
    fn from(left: RowMajorMatrix, right: ColMajorMatrix, output_matrix_order: u32) -> InData<'a> {
        assert!(left.ncols == right.nrows);
        assert!(output_matrix_order == 1 || output_matrix_order == 2);
        let mut squashed_data = Vec::<f32>::with_capacity(
            (left.get_n_elems() + right.get_n_elems())
                .try_into()
                .unwrap(),
        );
        squashed_data.extend(left.data.iter());
        squashed_data.extend(right.data.iter());
        InData {
            matrix1_ncols: left.ncols,
            matrix1_nrows: left.nrows,
            matrix2_ncols: right.ncols,
            // matrix2_nrows == matrix1_ncols,
            output_matrix_order,
            in_matrix_data: Cow::from(squashed_data),
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
        .open("shader-matrix-mult-simple.wgsl")
        .unwrap()
        .read_to_string(&mut cs_source)
        .unwrap();
    let cs_module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("Compute module"),
        source: wgpu::ShaderSource::Wgsl(Cow::from(cs_source)),
    });

    let mut rng = StdRng::from_entropy();
    let mut left_mat = RowMajorMatrix::new(4000, 4000);
    let mut right_mat = ColMajorMatrix::new(4000, 4000);

    for e in left_mat.data.iter_mut() {
        *e = rng.gen();
    }

    for e in right_mat.data.iter_mut() {
        *e = rng.gen();
    }

    let out_matrix_type = 2;
    let out_mat_nrows = left_mat.nrows;
    let out_mat_ncols = right_mat.ncols;
    assert!(left_mat.ncols == right_mat.nrows);

    println!(
        "Output will be {} rows x {} cols",
        out_mat_nrows, out_mat_ncols
    );

    let in_data = InData::from(left_mat.clone(), right_mat.clone(), out_matrix_type);

    let gpu_start = Instant::now();
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
        n_workgroups: usize::div_ceil(usize::try_from(out_mat_ncols * out_mat_nrows).unwrap(), 32),
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
    assert!(res.data.len() == usize::try_from(out_mat_nrows * out_mat_ncols).unwrap());
    let gpu_end = Instant::now();
    device.poll(wgpu::MaintainBase::Wait).panic_on_timeout();
    println!("GPU took: {} ms", (gpu_end - gpu_start).as_millis());

    let cpu_start = Instant::now();
    let reference_res = left_mat.mult(&right_mat);
    let cpu_end = Instant::now();
    println!("CPU took: {} ms", (cpu_end - cpu_start).as_millis());

    // Cross check result
    for i in 0..res.nrows {
        for j in 0..res.ncols {
            let i: usize = i.try_into().unwrap();
            let j: usize = j.try_into().unwrap();

            if (res[(i, j)] - reference_res[(i, j)]).abs() > 0.001 {
                panic!(
                    "Mismatch of results at {}, {}: gpu said: {}, cpu said: {}!",
                    i,
                    j,
                    res[(i, j)],
                    reference_res[(i, j)]
                );
            }
        }
    }
    println!("Ok!");

    println!(
        "GPU is {}% faster than CPU!",
        (1.0 - (gpu_end - gpu_start).as_secs_f32() / (cpu_end - cpu_start).as_secs_f32()) * 100.0
    );
}
