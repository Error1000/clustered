use core::fmt::Debug;
use core::ops::{Index, IndexMut};

#[macro_export]
macro_rules! matrix_impl {
    ($struct_name:ident) => {
        impl<T> Index<(usize, usize)> for $struct_name<T>
        where
            T: Clone + Default,
        {
            type Output = T;
            fn index(&self, index: (usize, usize)) -> &Self::Output {
                let off = self.index_to_offset(index);
                &self.data[off]
            }
        }

        impl<T> IndexMut<(usize, usize)> for $struct_name<T>
        where
            T: Clone + Default,
        {
            fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
                let off = self.index_to_offset(index);
                &mut self.data[off]
            }
        }

        impl<T> Debug for $struct_name<T>
        where
            T: Clone + Default + Debug,
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
pub struct ColMajorMatrix<T>
where
    T: Clone + Default,
{
    pub ncols: u32,
    pub nrows: u32,
    pub data: Vec<T>,
}

matrix_impl!(ColMajorMatrix);

impl<T> ColMajorMatrix<T>
where
    T: Clone + Default,
{
    pub fn new(nrows: u32, ncols: u32) -> Self {
        let mut inner = Vec::<T>::new();
        inner.resize(usize::try_from(ncols * nrows).unwrap(), T::default());
        Self {
            ncols,
            nrows,
            data: inner,
        }
    }

    pub fn index_to_offset(&self, index: (usize, usize)) -> usize {
        // i + j * number_of_elems_in_column
        index.0 + index.1 * (usize::try_from(self.nrows).unwrap())
    }

    pub fn nrows(&self) -> usize {
        self.nrows.try_into().unwrap()
    }

    pub fn ncols(&self) -> usize {
        self.ncols.try_into().unwrap()
    }

    pub fn get_n_elems(&self) -> usize {
        self.nrows() * self.ncols()
    }

    pub fn transpose_lazy(self) -> RowMajorMatrix<T> {
        RowMajorMatrix {
            nrows: self.ncols,
            ncols: self.nrows,
            data: self.data,
        }
    }
}

#[derive(Clone, PartialEq)]
pub struct RowMajorMatrix<T>
where
    T: Clone + Default,
{
    pub ncols: u32,
    pub nrows: u32,
    pub data: Vec<T>,
}

matrix_impl!(RowMajorMatrix);

impl<T> RowMajorMatrix<T>
where
    T: Clone + Default,
{
    pub fn new(nrows: u32, ncols: u32) -> Self {
        let mut inner = Vec::<T>::new();
        inner.resize(usize::try_from(ncols * nrows).unwrap(), T::default());
        Self {
            ncols,
            nrows,
            data: inner,
        }
    }

    pub fn index_to_offset(&self, index: (usize, usize)) -> usize {
        // j + i * number_of_elems_in_row
        index.1 + index.0 * (usize::try_from(self.ncols).unwrap())
    }

    pub fn nrows(&self) -> usize {
        self.nrows.try_into().unwrap()
    }

    pub fn ncols(&self) -> usize {
        self.ncols.try_into().unwrap()
    }

    pub fn get_n_elems(&self) -> usize {
        self.nrows() * self.ncols()
    }

    pub fn transpose_lazy(self) -> ColMajorMatrix<T> {
        ColMajorMatrix {
            ncols: self.nrows,
            nrows: self.ncols,
            data: self.data,
        }
    }
}
