use core::fmt::Debug;
use core::ops::{Index, IndexMut};

pub trait Matrix {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn index_to_offset(&self, index: (usize, usize)) -> usize;
}

#[macro_export]
macro_rules! matrix_derive {
    ($struct_name:ident) => {
        impl<MatrixElem> Index<(usize, usize)> for $struct_name<MatrixElem> {
            type Output = MatrixElem;
            fn index(&self, index: (usize, usize)) -> &Self::Output {
                let off = self.index_to_offset(index);
                &self.data[off]
            }
        }

        impl<MatrixElem> IndexMut<(usize, usize)> for $struct_name<MatrixElem> {
            fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
                let off = self.index_to_offset(index);
                &mut self.data[off]
            }
        }

        impl<MatrixElem> Debug for $struct_name<MatrixElem>
        where
            MatrixElem: Debug,
        {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                for i in 0..self.nrows() {
                    for j in 0..self.ncols() {
                        write!(f, "{:?} ", self[(i, j)])?;
                    }
                    writeln!(f)?;
                }
                Ok(())
            }
        }
    };
}

#[derive(Clone, PartialEq)]
pub struct ColMajorMatrix<MatrixElem> {
    pub ncols: u32,
    pub nrows: u32,
    pub data: Vec<MatrixElem>,
}

impl<MatrixElem> Matrix for ColMajorMatrix<MatrixElem> {
    fn index_to_offset(&self, index: (usize, usize)) -> usize {
        assert!(index.0 < self.nrows() && index.1 < self.ncols());
        // i + j * number_of_elems_in_column
        index.1 * self.nrows() + index.0
    }

    fn nrows(&self) -> usize {
        self.nrows.try_into().unwrap()
    }

    fn ncols(&self) -> usize {
        self.ncols.try_into().unwrap()
    }
}
matrix_derive!(ColMajorMatrix);

impl<MatrixElem> ColMajorMatrix<MatrixElem> {
    pub fn new(nrows: u32, ncols: u32) -> Self
    where
        MatrixElem: Default + Clone,
    {
        let mut inner = Vec::<MatrixElem>::new();
        inner.resize(
            usize::try_from(ncols * nrows).unwrap(),
            MatrixElem::default(),
        );
        Self {
            ncols,
            nrows,
            data: inner,
        }
    }

    pub fn get_n_elems(&self) -> usize {
        self.nrows() * self.ncols()
    }

    #[allow(dead_code)]
    pub fn transpose_lazy(self) -> RowMajorMatrix<MatrixElem> {
        RowMajorMatrix {
            nrows: self.ncols,
            ncols: self.nrows,
            data: self.data,
        }
    }
}

#[derive(Clone, PartialEq)]
pub struct RowMajorMatrix<MatrixElem> {
    pub ncols: u32,
    pub nrows: u32,
    pub data: Vec<MatrixElem>,
}

impl<MatrixElem> Matrix for RowMajorMatrix<MatrixElem> {
    fn index_to_offset(&self, index: (usize, usize)) -> usize {
        assert!(index.0 < self.nrows() && index.1 < self.ncols());
        // j + i * number_of_elems_in_row
        index.1 + index.0 * (usize::try_from(self.ncols).unwrap())
    }

    fn nrows(&self) -> usize {
        self.nrows.try_into().unwrap()
    }

    fn ncols(&self) -> usize {
        self.ncols.try_into().unwrap()
    }
}
matrix_derive!(RowMajorMatrix);

impl<MatrixElem> RowMajorMatrix<MatrixElem> {
    pub fn new(nrows: u32, ncols: u32) -> Self
    where
        MatrixElem: Default + Clone,
    {
        let mut inner = Vec::<MatrixElem>::new();
        inner.resize(
            usize::try_from(ncols * nrows).unwrap(),
            MatrixElem::default(),
        );
        Self {
            ncols,
            nrows,
            data: inner,
        }
    }

    pub fn get_n_elems(&self) -> usize {
        self.nrows() * self.ncols()
    }

    #[allow(dead_code)]
    pub fn transpose_lazy(self) -> ColMajorMatrix<MatrixElem> {
        ColMajorMatrix {
            ncols: self.nrows,
            nrows: self.ncols,
            data: self.data,
        }
    }
}
