#[path = "../bin-utils/matrix.rs"]
mod matrix;

use std::time::Instant;

use matrix::*;

use rand::{rngs::StdRng, Rng, SeedableRng};

#[allow(clippy::erasing_op, clippy::identity_op)]
pub fn mult(left: &RowMajorMatrix<f32>, right: &ColMajorMatrix<f32>) -> RowMajorMatrix<f32> {
    const CHUNK_SIZE: usize = 4;
    assert!(left.ncols == right.nrows);
    use rayon::prelude::*;
    RowMajorMatrix {
        nrows: left.nrows,
        ncols: right.ncols,
        data: (0..left.nrows())
            .into_par_iter()
            .flat_map(|i| {
                (0..right.ncols()).into_par_iter().map(move |j| {
                    [
                        (left.ncols() / CHUNK_SIZE * 0..left.ncols() / CHUNK_SIZE * 1),
                        (left.ncols() / CHUNK_SIZE * 1..left.ncols() / CHUNK_SIZE * 2),
                        (left.ncols() / CHUNK_SIZE * 2..left.ncols() / CHUNK_SIZE * 3),
                        (left.ncols() / CHUNK_SIZE * 3..left.ncols()),
                    ]
                    .into_par_iter()
                    .map(|subrange| {
                        subrange
                            .map(move |k| left[(i, k)] * right[(k, j)])
                            .sum::<f32>()
                    })
                    .sum()

                    // Old code for reference
                    // (0..left.ncols())
                    //     .map(move |k| left[(i, k)] * right[(k, j)])
                    //     .sum()
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
    let mut left_mat = RowMajorMatrix::new(400, 400);
    let mut right_mat = ColMajorMatrix::new(400, 400);
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

    let out_mat_nrows = left_mat.nrows();
    let out_mat_ncols = right_mat.ncols();
    assert!(left_mat.ncols == right_mat.nrows);
    println!(
        "Output will be {} cols x {} rows!",
        out_mat_ncols, out_mat_nrows
    );
    let time_start = Instant::now();
    #[allow(unused_variables)]
    let res = mult(&left_mat, &right_mat);
    let time_end = Instant::now();
    println!("{:?}", res);
    // println!("Took {} s", (time_end - time_start).as_secs_f64());
}
