use std::{ fmt::{ self, Formatter }, ops::{ Index, RangeFrom, RangeTo } };
use std::fmt::Debug;

#[derive(Clone)]
pub struct HStrides {
    pub(crate) strides: Vec<i64>,
    pub(crate) reduced_dim: usize,
    pub(crate) offset: i64,
}

impl Index<usize> for HStrides {
    type Output = i64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.strides[index]
    }
}

impl Index<RangeFrom<usize>> for HStrides {
    type Output = [i64];

    fn index(&self, index: RangeFrom<usize>) -> &Self::Output {
        &self.strides[index]
    }
}

impl Index<RangeTo<usize>> for HStrides {
    type Output = [i64];

    fn index(&self, index: RangeTo<usize>) -> &Self::Output {
        &self.strides[index]
    }
}

impl Debug for HStrides {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.strides)
    }
}
