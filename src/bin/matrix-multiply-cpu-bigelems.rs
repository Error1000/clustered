#[path = "../bin-utils/matrix.rs"]
mod matrix;

use matrix::*;
use std::array;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{self, Index, IndexMut};
use std::time::Instant;

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

macro_rules! calc_elem {
    ($i:literal, $j: literal, $res:ident, $left:ident, $right:ident) => {
        // Ugh, parentheses spaghetti
        $res[($i, $j)] = &(&(&(&$left[($i, 0)] * &$right[(0, $j)])
            + &(&$left[($i, 1)] * &$right[(1, $j)]))
            + &(&$left[($i, 2)] * &$right[(2, $j)]))
            + &(&$left[($i, 3)] * &$right[(3, $j)]);
    };
}
impl ops::Mul<&RowMajorMat4x4<f32>> for &RowMajorMat4x4<f32> {
    type Output = RowMajorMat4x4<f32>;

    fn mul(self, rhs: &RowMajorMat4x4<f32>) -> Self::Output {
        let mut res = RowMajorMat4x4 {
            data: array::from_fn(|_| f32::default()),
        };
        calc_elem!(0, 0, res, self, rhs);
        calc_elem!(0, 1, res, self, rhs);
        calc_elem!(0, 2, res, self, rhs);
        calc_elem!(0, 3, res, self, rhs);
        calc_elem!(1, 0, res, self, rhs);
        calc_elem!(1, 1, res, self, rhs);
        calc_elem!(1, 2, res, self, rhs);
        calc_elem!(1, 3, res, self, rhs);
        calc_elem!(2, 0, res, self, rhs);
        calc_elem!(2, 1, res, self, rhs);
        calc_elem!(2, 2, res, self, rhs);
        calc_elem!(2, 3, res, self, rhs);
        calc_elem!(3, 0, res, self, rhs);
        calc_elem!(3, 1, res, self, rhs);
        calc_elem!(3, 2, res, self, rhs);
        calc_elem!(3, 3, res, self, rhs);
        res
    }
}

impl Sum for RowMajorMat4x4<f32> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut res = RowMajorMat4x4 { data: [0f32; 16] };
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
fn mult(
    left: &RowMajorMatrix<RowMajorMat4x4<f32>>,
    right: &ColMajorMatrix<RowMajorMat4x4<f32>>,
) -> RowMajorMatrix<RowMajorMat4x4<f32>> {
    const CHUNK_SIZE: usize = 4;
    assert!(left.ncols == right.nrows);
    let inner_dim = left.ncols();

    use rayon::prelude::*;
    RowMajorMatrix {
        nrows: left.nrows,
        ncols: right.ncols,
        data: (0..left.nrows())
            .into_par_iter()
            .flat_map(|i| {
                (0..right.ncols()).into_par_iter().map(move |j| {
                    // Inner loop
                    [
                        (inner_dim / CHUNK_SIZE * 0..inner_dim / CHUNK_SIZE * 1),
                        (inner_dim / CHUNK_SIZE * 1..inner_dim / CHUNK_SIZE * 2),
                        (inner_dim / CHUNK_SIZE * 2..inner_dim / CHUNK_SIZE * 3),
                        (inner_dim / CHUNK_SIZE * 3..inner_dim),
                    ]
                    .into_par_iter()
                    .map(|subrange| {
                        subrange
                            .map(move |k| &left[(i, k)] * &right[(k, j)])
                            .sum::<RowMajorMat4x4<f32>>()
                    })
                    .sum()
                    // (0..inner_dim)
                    //     .map(move |k| &left[(i, k)] * &right[(k, j)])
                    //     .sum::<RowMajorMat4x4<f32>>()
                })
            })
            .collect(),
    }
}

#[tokio::main]
async fn main() {
    println!("Using CPU!");

    let mut buf = String::new();
    std::io::stdin().read_line(&mut buf).unwrap();
    let mut rng = StdRng::seed_from_u64(buf.trim().parse::<u64>().unwrap());
    drop(buf);
    // let mut rng = StdRng::from_entropy();
    // 4000x4000 square matrix multiplication performance measurement (31/aug/2024): ~850 ms
    let mut left_mat = RowMajorMatrix::<RowMajorMat4x4<f32>>::new(4000 / 4, 4000 / 4);
    let mut right_mat = ColMajorMatrix::<RowMajorMat4x4<f32>>::new(4000 / 4, 4000 / 4);
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

    let out_mat_nrows = left_mat.nrows();
    let out_mat_ncols = right_mat.ncols();
    assert!(left_mat.ncols == right_mat.nrows);
    println!(
        "Output will be {} cols x {} rows!",
        out_mat_ncols * 4,
        out_mat_nrows * 4
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
