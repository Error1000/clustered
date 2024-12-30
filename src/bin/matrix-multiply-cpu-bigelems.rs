#[path = "../bin-utils/matrix.rs"]
mod matrix;

use matrix::*;
use std::array;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{self, Index, IndexMut};
use std::time::Instant;

const TILE_SIZE: usize = 8;
type TypeASub = RowMajorMatTile<f32>;
type TypeA = RowMajorMatrix<TypeASub>;

type TypeBSub = ColMajorMatTile<f32>;
type TypeB = ColMajorMatrix<TypeBSub>;

type TypeCSub = ColMajorMatTile<f32>;
type TypeC = ColMajorMatrix<TypeCSub>;

use rand::{rngs::StdRng, Rng, SeedableRng};

#[derive(Clone)]
struct RowMajorMatTile<MatrixElem> {
    data: [MatrixElem; TILE_SIZE * TILE_SIZE],
}

impl<MatrixElem> Default for RowMajorMatTile<MatrixElem>
where
    MatrixElem: Default,
{
    fn default() -> Self {
        Self {
            data: array::from_fn(|_| Default::default()),
        }
    }
}

#[derive(Clone)]
struct ColMajorMatTile<MatrixElem> {
    data: [MatrixElem; TILE_SIZE * TILE_SIZE],
}

impl<MatrixElem> Default for ColMajorMatTile<MatrixElem>
where
    MatrixElem: Default,
{
    fn default() -> Self {
        Self {
            data: array::from_fn(|_| Default::default()),
        }
    }
}

impl<MatrixElem> RowMajorMatTile<MatrixElem> {
    fn nrows(&self) -> usize {
        TILE_SIZE
    }
    fn ncols(&self) -> usize {
        TILE_SIZE
    }
    fn index_to_offset(&self, index: (usize, usize)) -> usize {
        assert!(index.0 < TILE_SIZE && index.1 < TILE_SIZE);
        index.0 * TILE_SIZE + index.1
    }
}

matrix_derive!(RowMajorMatTile);

impl<MatrixElem> ColMajorMatTile<MatrixElem> {
    fn nrows(&self) -> usize {
        TILE_SIZE
    }
    fn ncols(&self) -> usize {
        TILE_SIZE
    }
    fn index_to_offset(&self, index: (usize, usize)) -> usize {
        assert!(index.0 < TILE_SIZE && index.1 < TILE_SIZE);
        index.1 * TILE_SIZE + index.0
    }
}
matrix_derive!(ColMajorMatTile);

impl ops::Mul<&TypeBSub> for &TypeASub {
    type Output = TypeCSub;

    fn mul(self, rhs: &TypeBSub) -> Self::Output {
        let mut res = TypeCSub {
            data: array::from_fn(|_| f32::default()),
        };
        for i in 0..TILE_SIZE {
            for j in 0..TILE_SIZE {
                let mut elem = 0f32;
                for k in 0..TILE_SIZE {
                    elem += self[(i, k)] * rhs[(k, j)];
                }
                res[(i, j)] = elem;
            }
        }
        res
    }
}

impl Sum for TypeCSub {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut res = TypeCSub {
            data: array::from_fn(|_| f32::default()),
        };
        for e in iter {
            for i in 0..res.data.len() {
                res.data[i] += e.data[i];
            }
        }
        res
    }
}

// fn index_to_offset(&self, index: (usize, usize)) -> usize {
//     assert!(index.0 < self.nrows() && index.1 < self.ncols());
//     let matrix_row_size = self.ncols() / self.chunk_ncols();
//     let bigelem_row_size = self.chunk_ncols();
//     // Find big element index
//     let big_elem_index = (
//         index.0 / self.bigelem_nrows(),
//         index.1 / self.bigelem_ncols(),
//     );
//     let big_elem_start_offset = (big_elem_index.0 * matrix_row_size + big_elem_index.1)
//         * (self.bigelem_nrows() * self.bigelem_ncols());
//     let sub_elem_index = (
//         index.0 % self.bigelem_nrows(),
//         index.1 % self.bigelem_ncols(),
//     );
//     let sub_elem_extra_offset = sub_elem_index.0 * bigelem_row_size + sub_elem_index.1;
//     big_elem_start_offset + sub_elem_extra_offset
// }

#[allow(clippy::erasing_op, clippy::identity_op)]
fn mult(left: &TypeA, right: &TypeB) -> TypeC {
    const CHUNK_SIZE: usize = 4;
    assert!(left.ncols == right.nrows);
    let inner_dim = left.ncols();

    use rayon::prelude::*;
    TypeC {
        nrows: left.nrows,
        ncols: right.ncols,
        data: (0..left.nrows())
            .into_par_iter()
            .flat_map(|i| {
                (0..right.ncols()).into_par_iter().map(move |j| {
                    // Inner loop
                    // [
                    //     (inner_dim / CHUNK_SIZE * 0..inner_dim / CHUNK_SIZE * 1),
                    //     (inner_dim / CHUNK_SIZE * 1..inner_dim / CHUNK_SIZE * 2),
                    //     (inner_dim / CHUNK_SIZE * 2..inner_dim / CHUNK_SIZE * 3),
                    //     (inner_dim / CHUNK_SIZE * 3..inner_dim),
                    // ]
                    // .into_par_iter()
                    // .map(|subrange| {
                    //     subrange
                    //         .map(move |k| &left[(i, k)] * &right[(k, j)])
                    //         .sum::<TypeCSub>()
                    // })
                    // .sum()
                    (0..inner_dim)
                        .map(move |k| &left[(i, k)] * &right[(k, j)])
                        .sum::<TypeCSub>()
                })
            })
            .collect(),
    }
}

#[tokio::main]
async fn main() {
    println!("Using CPU!");
    println!("Tile size: {TILE_SIZE}!");
    let mut buf = String::new();
    std::io::stdin().read_line(&mut buf).unwrap();
    let mut rng = StdRng::seed_from_u64(buf.trim().parse::<u64>().unwrap());
    drop(buf);
    // let mut rng = StdRng::from_entropy();
    // 4000x4000 square matrix multiplication performance measurement (31/aug/2024): ~850 ms
    let side_len = u32::try_from(4096 / TILE_SIZE).unwrap();
    assert!(4096 % TILE_SIZE == 0);
    let mut left_mat = TypeA::new(side_len, side_len);
    let mut right_mat = TypeB::new(side_len, side_len);
    for i in 0..left_mat.nrows() * TILE_SIZE {
        for j in 0..left_mat.ncols() * TILE_SIZE {
            left_mat[(i / TILE_SIZE, j / TILE_SIZE)][(i % TILE_SIZE, j % TILE_SIZE)] = rng.gen();
        }
    }

    for i in 0..right_mat.nrows() * TILE_SIZE {
        for j in 0..right_mat.ncols() * TILE_SIZE {
            right_mat[(i / TILE_SIZE, j / TILE_SIZE)][(i % TILE_SIZE, j % TILE_SIZE)] = rng.gen();
        }
    }

    let out_mat_nrows = left_mat.nrows();
    let out_mat_ncols = right_mat.ncols();
    assert!(left_mat.ncols == right_mat.nrows);
    println!(
        "Output will be {} cols x {} rows!",
        out_mat_ncols * TILE_SIZE,
        out_mat_nrows * TILE_SIZE
    );

    let time_start = Instant::now();
    let res = mult(&left_mat, &right_mat);
    let time_end = Instant::now();
    // for i in 0..res.nrows() * 4 {
    //     for j in 0..res.ncols() * 4 {
    //         print!("{:?} ", res[(i / 4, j / 4)][(i % 4, j % 4)]);
    //     }
    //     println!();
    // }

    println!("Took {} s", (time_end - time_start).as_secs_f64());
}
