use core::panic::Location;

use crate::err_handler::ErrHandler;

pub struct Axis {
    pub axes: Vec<i64>,
}

#[cfg_attr(feature = "track_caller", track_caller)]
pub fn process_axes<T: Into<Axis>>(axes: T, ndim: usize) -> anyhow::Result<Vec<usize>> {
    let ndim = ndim as i64;
    let axes = axes.into().axes;
    let mut new_axes = Vec::with_capacity(axes.len());
    for &axis in axes.iter() {
        if axis < 0 {
            let val = axis + ndim;
            new_axes.push(val as usize);
        } else {
            if axis >= ndim {
                return Err(ErrHandler::IndexOutOfRange(ndim as usize, axis, axis + ndim, Location::caller()).into());
            }
            new_axes.push(axis as usize);
        }
    }
    Ok(new_axes)
}

impl<'a> From<&'a [i64]> for Axis {
    fn from(axes: &'a [i64]) -> Self {
        Axis { axes: axes.to_vec() }
    }
}

impl<'a> From<&'a [i32]> for Axis {
    fn from(axes: &'a [i32]) -> Self {
        Axis { axes: axes.iter().map(|x| *x as i64).collect() }
    }
}

impl<'a> From<&'a [i16]> for Axis {
    fn from(axes: &'a [i16]) -> Self {
        Axis { axes: axes.iter().map(|x| *x as i64).collect() }
    }
}

impl<'a> From<&'a [i8]> for Axis {
    fn from(axes: &'a [i8]) -> Self {
        Axis { axes: axes.iter().map(|x| *x as i64).collect() }
    }
}

impl<'a> From<&'a [i128]> for Axis {
    fn from(axes: &'a [i128]) -> Self {
        Axis { axes: axes.iter().map(|x| *x as i64).collect() }
    }
}

impl<'a, const N: usize> From<&'a [i64; N]> for Axis {
    fn from(axes: &'a [i64; N]) -> Self {
        Axis { axes: axes.to_vec() }
    }
}

impl<'a, const N: usize> From<&'a [i32; N]> for Axis {
    fn from(axes: &'a [i32; N]) -> Self {
        Axis { axes: axes.iter().map(|x| *x as i64).collect() }
    }
}

impl<'a, const N: usize> From<&'a [i16; N]> for Axis {
    fn from(axes: &'a [i16; N]) -> Self {
        Axis { axes: axes.iter().map(|x| *x as i64).collect() }
    }
}

impl<'a, const N: usize> From<&'a [i8; N]> for Axis {
    fn from(axes: &'a [i8; N]) -> Self {
        Axis { axes: axes.iter().map(|x| *x as i64).collect() }
    }
}

impl<'a, const N: usize> From<&'a [i128; N]> for Axis {
    fn from(axes: &'a [i128; N]) -> Self {
        Axis { axes: axes.iter().map(|x| *x as i64).collect() }
    }
}

impl<const N: usize> From<[i64; N]> for Axis {
    fn from(axes: [i64; N]) -> Self {
        Axis { axes: axes.to_vec() }
    }
}

impl From<Vec<i64>> for Axis {
    fn from(axes: Vec<i64>) -> Self {
        Axis { axes }
    }
}

impl<'a> From<&'a Vec<i64>> for Axis {
    fn from(axes: &'a Vec<i64>) -> Self {
        Axis { axes: axes.clone() }
    }
}

impl<'a> From<&'a [usize]> for Axis {
    fn from(axes: &'a [usize]) -> Self {
        Axis {
            axes: axes
                .iter()
                .map(|x| *x as i64)
                .collect(),
        }
    }
}

impl From<Vec<usize>> for Axis {
    fn from(axes: Vec<usize>) -> Self {
        Axis {
            axes: axes
                .iter()
                .map(|x| *x as i64)
                .collect(),
        }
    }
}

impl From<i64> for Axis {
    fn from(axes: i64) -> Self {
        Axis { axes: vec![axes] }
    }
}
