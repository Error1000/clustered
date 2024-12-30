#[path = "../bin-utils/matrix.rs"]
mod matrix;
use std::{
    fmt::Debug,
    fs::OpenOptions,
    io::Read,
    net::{Ipv4Addr, SocketAddrV4},
    ops::{Index, IndexMut},
    time::Instant,
};

use clustered::serialisable_program::SerialisableProgram;
use matrix::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use tokio::net::TcpStream;

// 3 memory levels, 3 matrix sizes
// register level -> (nrows_left = innersize = ncols_right = one value), this is the small tile
// warp/shared memory level -> (nrows_left, innersize, ncols_right), this is the big tile
// global memory level -> (nrows_left, innersize, ncols_right), this is the whole two matrices
const SMALL_TILE_N: usize = 4;

const BIG_TILE_NROW_LEFT: usize = 1;
const BIG_TILE_INNER_SIZE: usize = 1;
const BIG_TILE_NCOL_RIGHT: usize = 1;

// These are column major because that's what the shader memory layout uses
// so when we serialise the array it will be in the right order for the shader
// as the serailisation code is (intentionally dumb)/simple and doesn't know much about what
// the right format is beyong basic padding and primitives
#[derive(Default, Clone, Copy)]
struct ColMajorSmallTile<MatrixElem> {
    data: [MatrixElem; SMALL_TILE_N * SMALL_TILE_N],
}

impl<MatrixElem> ColMajorSmallTile<MatrixElem> {
    fn nrows(&self) -> usize {
        SMALL_TILE_N
    }
    fn ncols(&self) -> usize {
        SMALL_TILE_N
    }
    fn index_to_offset(&self, index: (usize, usize)) -> usize {
        assert!(index.0 < self.nrows() && index.1 < self.ncols());
        let n_elems_per_col = self.nrows();
        index.0 + index.1 * n_elems_per_col
    }
}
matrix_derive!(ColMajorSmallTile);

// For now we only support row majort big tiles
#[derive(Default, Clone)]
struct RowMajorBigTileLeft<MatrixElem> {
    data: [MatrixElem; BIG_TILE_NROW_LEFT * BIG_TILE_INNER_SIZE],
}
impl<MatrixElem> RowMajorBigTileLeft<MatrixElem> {
    fn nrows(&self) -> usize {
        BIG_TILE_NROW_LEFT
    }
    fn ncols(&self) -> usize {
        BIG_TILE_INNER_SIZE
    }
    fn index_to_offset(&self, index: (usize, usize)) -> usize {
        assert!(index.0 < self.nrows() && index.1 < self.ncols());
        let n_elems_per_row = self.ncols();
        index.0 * n_elems_per_row + index.1
    }
}
matrix_derive!(RowMajorBigTileLeft);

#[derive(Default, Clone)]
struct RowMajorBigTileRight<MatrixElem> {
    data: [MatrixElem; BIG_TILE_INNER_SIZE * BIG_TILE_NCOL_RIGHT],
}

impl<MatrixElem> RowMajorBigTileRight<MatrixElem> {
    fn nrows(&self) -> usize {
        BIG_TILE_INNER_SIZE
    }
    fn ncols(&self) -> usize {
        BIG_TILE_NCOL_RIGHT
    }
    fn index_to_offset(&self, index: (usize, usize)) -> usize {
        assert!(index.0 < self.nrows() && index.1 < self.ncols());
        let n_elems_per_row = self.ncols();
        index.0 * n_elems_per_row + index.1
    }
}
matrix_derive!(RowMajorBigTileRight);

#[derive(Default, Clone)]
struct RowMajorBigTileOutput<MatrixElem> {
    data: [MatrixElem; BIG_TILE_NROW_LEFT * BIG_TILE_NCOL_RIGHT],
}

impl<MatrixElem> RowMajorBigTileOutput<MatrixElem> {
    fn nrows(&self) -> usize {
        BIG_TILE_NROW_LEFT
    }
    fn ncols(&self) -> usize {
        BIG_TILE_NCOL_RIGHT
    }
    fn index_to_offset(&self, index: (usize, usize)) -> usize {
        assert!(index.0 < self.nrows() && index.1 < self.ncols());
        let n_elems_per_row = self.ncols();
        index.0 * n_elems_per_row + index.1
    }
}
matrix_derive!(RowMajorBigTileOutput);

struct InData {
    inner_dim: u32,
    nrows_left: u32,
    ncols_right: u32,
    output_matrix_order: u32,
    packed_data: Vec<ColMajorSmallTile<f32>>,
}

impl InData {
    fn from(
        left_mat: &RowMajorMatrix<RowMajorBigTileLeft<ColMajorSmallTile<f32>>>,
        right_mat: &RowMajorMatrix<RowMajorBigTileRight<ColMajorSmallTile<f32>>>,
        output_matrix_order: u32,
    ) -> Self {
        assert!(left_mat.ncols == right_mat.nrows);

        let mut flattened_matrix_data = Vec::new();
        flattened_matrix_data.extend(
            left_mat
                .data
                .iter()
                .flat_map(|elem| elem.data.iter())
                .cloned(),
        );
        flattened_matrix_data.extend(
            right_mat
                .data
                .iter()
                .flat_map(|elem| elem.data.iter())
                .cloned(),
        );

        InData {
            inner_dim: left_mat.ncols,
            nrows_left: left_mat.nrows,
            ncols_right: right_mat.ncols,
            output_matrix_order,
            packed_data: flattened_matrix_data,
        }
    }

    fn into_shader_bytes(self) -> Vec<u8> {
        let mut res = Vec::new();
        res.extend(self.inner_dim.to_le_bytes());
        res.extend(self.nrows_left.to_le_bytes());
        res.extend(self.ncols_right.to_le_bytes());
        res.extend(self.output_matrix_order.to_le_bytes());
        res.extend(self.packed_data.into_iter().flat_map(|small_tile| {
            small_tile
                .data
                .into_iter()
                .flat_map(|value| value.to_le_bytes().into_iter())
        }));
        res
    }
}

#[tokio::main]
async fn main() {
    let mut buf = String::new();
    std::io::stdin().read_line(&mut buf).unwrap();
    let mut rng = StdRng::seed_from_u64(buf.trim().parse::<u64>().unwrap());
    drop(buf);
    //    let mut rng = StdRng::from_entropy();

    let mut left_mat = RowMajorMatrix::<RowMajorBigTileLeft<ColMajorSmallTile<f32>>>::new(
        u32::try_from(4 / BIG_TILE_NROW_LEFT / SMALL_TILE_N).unwrap(),
        u32::try_from(4 / BIG_TILE_INNER_SIZE / SMALL_TILE_N).unwrap(),
    );
    let mut right_mat = RowMajorMatrix::<RowMajorBigTileRight<ColMajorSmallTile<f32>>>::new(
        u32::try_from(4 / BIG_TILE_INNER_SIZE / SMALL_TILE_N).unwrap(),
        u32::try_from(4 / BIG_TILE_NCOL_RIGHT / SMALL_TILE_N).unwrap(),
    );
    assert!(left_mat.ncols == right_mat.nrows);
    for i in 0..left_mat.nrows() * SMALL_TILE_N * BIG_TILE_NROW_LEFT {
        for j in 0..left_mat.ncols() * SMALL_TILE_N * BIG_TILE_INNER_SIZE {
            let big_tile_i = i / SMALL_TILE_N / BIG_TILE_NROW_LEFT;
            let big_tile_j = j / SMALL_TILE_N / BIG_TILE_INNER_SIZE;
            let small_tile_i = (i / SMALL_TILE_N) % BIG_TILE_NROW_LEFT;
            let small_tile_j = (j / SMALL_TILE_N) % BIG_TILE_INNER_SIZE;
            let elem_i = i % SMALL_TILE_N;
            let elem_j = j % SMALL_TILE_N;
            left_mat[(big_tile_i, big_tile_j)][(small_tile_i, small_tile_j)][(elem_i, elem_j)] =
                rng.gen();
        }
    }

    for i in 0..right_mat.nrows() * SMALL_TILE_N * BIG_TILE_INNER_SIZE {
        for j in 0..right_mat.ncols() * SMALL_TILE_N * BIG_TILE_NCOL_RIGHT {
            let big_tile_i = i / SMALL_TILE_N / BIG_TILE_INNER_SIZE;
            let big_tile_j = j / SMALL_TILE_N / BIG_TILE_NCOL_RIGHT;
            let small_tile_i = (i / SMALL_TILE_N) % BIG_TILE_INNER_SIZE;
            let small_tile_j = (j / SMALL_TILE_N) % BIG_TILE_NCOL_RIGHT;
            let elem_i = i % SMALL_TILE_N;
            let elem_j = j % SMALL_TILE_N;

            right_mat[(big_tile_i, big_tile_j)][(small_tile_i, small_tile_j)][(elem_i, elem_j)] =
                rng.gen();
        }
    }

    /*
    for i in 0..left_mat.nrows() * SMALL_TILE_N * BIG_TILE_NROW_LEFT {
        for j in 0..left_mat.ncols() * SMALL_TILE_N * BIG_TILE_INNER_SIZE {
            let big_tile_i = i / SMALL_TILE_N / BIG_TILE_NROW_LEFT;
            let big_tile_j = j / SMALL_TILE_N / BIG_TILE_INNER_SIZE;
            let small_tile_i = (i / SMALL_TILE_N) % BIG_TILE_NROW_LEFT;
            let small_tile_j = (j / SMALL_TILE_N) % BIG_TILE_INNER_SIZE;
            let elem_i = i % SMALL_TILE_N;
            let elem_j = j % SMALL_TILE_N;
            print!(
                "{:?} ",
                left_mat[(big_tile_i, big_tile_j)][(small_tile_i, small_tile_j)][(elem_i, elem_j)]
            );
        }
        println!();
    }

    for i in 0..right_mat.nrows() * SMALL_TILE_N * BIG_TILE_INNER_SIZE {
        for j in 0..right_mat.ncols() * SMALL_TILE_N * BIG_TILE_NCOL_RIGHT {
            let big_tile_i = i / SMALL_TILE_N / BIG_TILE_INNER_SIZE;
            let big_tile_j = j / SMALL_TILE_N / BIG_TILE_NCOL_RIGHT;
            let small_tile_i = (i / SMALL_TILE_N) % BIG_TILE_INNER_SIZE;
            let small_tile_j = (j / SMALL_TILE_N) % BIG_TILE_NCOL_RIGHT;
            let elem_i = i % SMALL_TILE_N;
            let elem_j = j % SMALL_TILE_N;
            print!(
                "{:?} ",
                right_mat[(big_tile_i, big_tile_j)][(small_tile_i, small_tile_j)][(elem_i, elem_j)]
            );
        }
        println!();
    } */

    let out_mat_nrows = left_mat.nrows();
    let out_mat_ncols = right_mat.ncols();
    println!(
        "Output will be {} cols x {} rows!",
        out_mat_ncols * SMALL_TILE_N * BIG_TILE_NCOL_RIGHT,
        out_mat_nrows * SMALL_TILE_N * BIG_TILE_NROW_LEFT
    );

    let time_start = Instant::now();

    let mut telefork_server_stream =
        TcpStream::connect(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 1337))
            .await
            .unwrap();

    let mut program_file = OpenOptions::new()
        .read(true)
        .create(false)
        .append(false)
        .write(false)
        .open("shader-matrix-mult-nested-bigelems.wgsl")
        .expect("Program file should exist!");
    let mut program_string = String::new();
    program_file.read_to_string(&mut program_string).unwrap();
    drop(program_file);

    let out_matrix_type = 1;
    let program_input_data = InData::from(&left_mat, &right_mat, out_matrix_type);
    let program_capsule = SerialisableProgram {
        entry_point: "main".to_owned(),
        program: program_string,
        n_workgroups: out_mat_ncols * out_mat_nrows,
        workgroup_size: BIG_TILE_NROW_LEFT * BIG_TILE_NCOL_RIGHT,
        in_data: program_input_data.into_shader_bytes(),
        out_data_nbytes: core::mem::size_of::<f32>()
            * out_mat_ncols
            * out_mat_nrows
            * BIG_TILE_NROW_LEFT
            * BIG_TILE_NCOL_RIGHT
            * SMALL_TILE_N
            * SMALL_TILE_N,
    };
    let serialised_program = serde_json::to_string(&program_capsule).unwrap();

    clustered::networking::write_buf(&mut telefork_server_stream, serialised_program.as_bytes())
        .await
        .unwrap();

    let raw_res = clustered::networking::read_buf(&mut telefork_server_stream)
        .await
        .unwrap();

    assert!(out_matrix_type == 1);
    assert!(raw_res.len() == program_capsule.out_data_nbytes);
    let res = ColMajorMatrix::<RowMajorBigTileOutput<ColMajorSmallTile<f32>>> {
        nrows: out_mat_nrows.try_into().unwrap(),
        ncols: out_mat_ncols.try_into().unwrap(),
        data: raw_res
            .chunks_exact(
                core::mem::size_of::<f32>()
                    * SMALL_TILE_N
                    * SMALL_TILE_N
                    * BIG_TILE_NROW_LEFT
                    * BIG_TILE_NCOL_RIGHT,
            )
            .map(|raw_big_tile| {
                let mut res_big_tile = RowMajorBigTileOutput::<ColMajorSmallTile<f32>> {
                    data: [ColMajorSmallTile::<f32> {
                        data: [0f32; SMALL_TILE_N * SMALL_TILE_N],
                    }; BIG_TILE_NCOL_RIGHT * BIG_TILE_NROW_LEFT],
                };
                for (i, raw_small_tile) in raw_big_tile
                    .chunks_exact(core::mem::size_of::<f32>() * SMALL_TILE_N * SMALL_TILE_N)
                    .enumerate()
                {
                    for (j, val) in raw_small_tile
                        .chunks_exact(core::mem::size_of::<f32>())
                        .map(|value_bytes| f32::from_le_bytes(value_bytes.try_into().unwrap()))
                        .enumerate()
                    {
                        res_big_tile.data[i].data[j] = val;
                    }
                }
                res_big_tile
            })
            .collect::<Vec<RowMajorBigTileOutput<ColMajorSmallTile<f32>>>>(),
    };
    let time_end = Instant::now();
    assert!(res.data.len() == out_mat_nrows * out_mat_ncols);
    println!("Took {}s!", (time_end - time_start).as_secs_f64());
    for i in 0..res.nrows() * SMALL_TILE_N * BIG_TILE_NROW_LEFT {
        for j in 0..res.ncols() * SMALL_TILE_N * BIG_TILE_NCOL_RIGHT {
            let big_tile_i = i / SMALL_TILE_N / BIG_TILE_NROW_LEFT;
            let big_tile_j = j / SMALL_TILE_N / BIG_TILE_NCOL_RIGHT;
            let small_tile_i = (i / SMALL_TILE_N) % BIG_TILE_NROW_LEFT;
            let small_tile_j = (j / SMALL_TILE_N) % BIG_TILE_NCOL_RIGHT;
            let elem_i = i % SMALL_TILE_N;
            let elem_j = j % SMALL_TILE_N;
            print!(
                "{:?} ",
                res[(big_tile_i, big_tile_j)][(small_tile_i, small_tile_j)][(elem_i, elem_j)]
            );
        }
        println!();
    }
}
