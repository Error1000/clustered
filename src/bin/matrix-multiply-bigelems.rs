#[path = "../bin-utils/matrix.rs"]
mod matrix;
use matrix::*;
use tokio::net::TcpStream;

use std::{
    borrow::Cow,
    fmt::Debug,
    fs::OpenOptions,
    io::{Read, Write},
    net::{Ipv4Addr, SocketAddrV4},
    ops::{Index, IndexMut},
    str::FromStr,
    time::Instant,
};

use clustered::serialisable_program::SerialisableProgram;
use rand::{rngs::StdRng, Rng, SeedableRng};

#[derive(Clone, Default)]
struct RowMajorMat4x4<MatrixElem> {
    data: [MatrixElem; 4 * 4],
}

#[derive(Clone, Default)]
struct ColMajorMat4x4<MatrixElem> {
    data: [MatrixElem; 4 * 4],
}

impl<MatrixElem> RowMajorMat4x4<MatrixElem> {
    fn nrows(&self) -> usize {
        4
    }
    fn ncols(&self) -> usize {
        4
    }
    fn index_to_offset(&self, index: (usize, usize)) -> usize {
        assert!(index.0 < 4 && index.1 < 4);
        index.0 * 4 + index.1
    }
}
matrix_impl!(RowMajorMat4x4);

impl<MatrixElem> ColMajorMat4x4<MatrixElem> {
    fn nrows(&self) -> usize {
        4
    }
    fn ncols(&self) -> usize {
        4
    }
    fn index_to_offset(&self, index: (usize, usize)) -> usize {
        assert!(index.0 < 4 && index.1 < 4);
        index.1 * 4 + index.0
    }
}
matrix_impl!(ColMajorMat4x4);

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
        left: &ColMajorMatrix<ColMajorMat4x4<f32>>,
        right: &RowMajorMatrix<ColMajorMat4x4<f32>>,
        output_matrix_order: u32,
    ) -> InData<'a> {
        assert!(left.ncols == right.nrows);
        assert!(output_matrix_order == 1 || output_matrix_order == 2);
        let mut formatted_data =
            Vec::<f32>::with_capacity(left.get_n_elems() + right.get_n_elems());
        formatted_data.extend(left.data.iter().flat_map(|elem| elem.data.into_iter()));
        formatted_data.extend(right.data.iter().flat_map(|elem| elem.data.into_iter()));
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
    let mut cs_source = String::new();
    OpenOptions::new()
        .read(true)
        .write(false)
        .open("shader-matrix-mult-bigelems.wgsl")
        .unwrap()
        .read_to_string(&mut cs_source)
        .unwrap();

    // let mut buf = String::new();
    // std::io::stdin().read_line(&mut buf).unwrap();
    // let mut rng = StdRng::seed_from_u64(buf.trim().parse::<u64>().unwrap());
    // drop(buf);
    let mut rng = StdRng::from_entropy();

    // According to the wgsl specs, section 16.1.2.14, matrix variables are column major
    let mut left_mat = ColMajorMatrix::<ColMajorMat4x4<f32>>::new(4000 / 4, 4000 / 4);
    let mut right_mat = RowMajorMatrix::<ColMajorMat4x4<f32>>::new(4000 / 4, 4000 / 4);

    for i in 0..left_mat.nrows() * 4 {
        for j in 0..left_mat.ncols() * 4 {
            left_mat[(i / 4, j / 4)][(i % 4, j % 4)] = rng.gen();
        }
    }

    for i in 0..right_mat.nrows() * 4 {
        for j in 0..right_mat.ncols() * 4 {
            right_mat[(i / 4, j / 4)][(i % 4, j % 4)] = rng.gen();
        }
    }

    let out_matrix_type = 1;
    let out_mat_nrows = left_mat.nrows;
    let out_mat_ncols = right_mat.ncols;
    println!(
        "Output will be {} cols x {} rows!",
        out_mat_ncols * 4,
        out_mat_nrows * 4
    );

    let mut telefork_server_stream =
        TcpStream::connect(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 1337))
            .await
            .unwrap();

    let time_start = Instant::now();
    assert!(left_mat.ncols == right_mat.nrows);
    let in_data = InData::from(&left_mat, &right_mat, out_matrix_type);

    let program_capsule = SerialisableProgram {
        in_data: in_data.into_shader_bytes(),
        out_data_nbytes: core::mem::size_of::<f32>()
            * usize::try_from(out_mat_ncols * out_mat_nrows * 4 * 4).unwrap(),
        program: cs_source,
        entry_point: "main".to_owned(),
        n_workgroups: usize::div_ceil(usize::try_from(out_mat_ncols * out_mat_nrows).unwrap(), 32),
        workgroup_size: 32,
    };
    let serialised_program = serde_json::to_string(&program_capsule).unwrap();
    // let mut program_file = OpenOptions::new()
    //     .create(true)
    //     .truncate(true)
    //     .write(true)
    //     .open("program-capsule.json")
    //     .unwrap();
    // program_file
    //     .write_all(serialised_program.as_bytes())
    //     .unwrap();
    // drop(program_file);

    clustered::networking::write_buf(&mut telefork_server_stream, serialised_program.as_bytes())
        .await
        .unwrap();

    let raw_res = clustered::networking::read_buf(&mut telefork_server_stream)
        .await
        .unwrap();

    assert!(out_matrix_type == 1);
    let res = ColMajorMatrix::<ColMajorMat4x4<f32>> {
        nrows: out_mat_nrows,
        ncols: out_mat_ncols,
        data: raw_res
            .chunks_exact(core::mem::size_of::<f32>() * 4 * 4)
            .map(|raw_elem| {
                let mut res_elem = ColMajorMat4x4 {
                    data: [0f32; 4 * 4],
                };
                for (i, val) in raw_elem
                    .chunks_exact(core::mem::size_of::<f32>())
                    .map(|value_bytes| f32::from_le_bytes(value_bytes.try_into().unwrap()))
                    .enumerate()
                {
                    res_elem.data[i] = val;
                }
                res_elem
            })
            .collect::<Vec<ColMajorMat4x4<f32>>>(),
    };
    let time_end = Instant::now();
    assert!(res.data.len() == usize::try_from(out_mat_nrows * out_mat_ncols).unwrap());
    println!("Took {}s!", (time_end - time_start).as_secs_f64());
    // for i in 0..res.nrows() * 4 {
    //     for j in 0..res.ncols() * 4 {
    //         print!("{:?} ", res[(i / 4, j / 4)][(i % 4, j % 4)]);
    //     }
    //     println!();
    // }
}
