use core::panic::Location;

use crate::err_handler::ErrHandler;

/// `Axis` struct to hold the axes for operations
///
/// it stores the axes the user wants to perform operations on
pub struct Axis {
    /// the axes to be processed
    pub axes: Vec<i64>,
}

/// function to process the axes
///
/// user could pass negative values for the axes, this function will convert them to positive values
///
/// # Arguments
///
/// * `axes` - the axes to be processed
///
/// * `ndim` - the number of dimensions of the tensor
///
/// # Returns
///
/// - the processed axes
///
/// # Error
///
/// if the axis is out of `0..ndim`
#[cfg_attr(feature = "track_caller", track_caller)]
pub fn process_axes<T: Into<Axis>>(axes: T, ndim: usize) -> anyhow::Result<Vec<usize>> {
    let ndim = ndim as i64;
    let axes = axes.into().axes;
    let mut new_axes = Vec::with_capacity(axes.len());
    for &axis in axes.iter() {
        if axis < 0 {
            let val = axis + ndim;
            if val < 0 {
                return Err(ErrHandler::IndexOutOfRangeCvt(
                    ndim as usize,
                    axis,
                    val,
                    Location::caller(),
                )
                .into());
            }
            new_axes.push(val as usize);
        } else {
            if axis >= ndim {
                return Err(
                    ErrHandler::IndexOutOfRange(ndim as usize, axis, Location::caller()).into(),
                );
            }
            new_axes.push(axis as usize);
        }
    }
    Ok(new_axes)
}

// Implementing the From trait for the `Axis` struct, when the user pass any of the following types, it will be converted to Axis automatically

impl<'a> From<&'a [i64]> for Axis {
    fn from(axes: &'a [i64]) -> Self {
        Axis {
            axes: axes.to_vec(),
        }
    }
}

impl<'a> From<&'a [i32]> for Axis {
    fn from(axes: &'a [i32]) -> Self {
        Axis {
            axes: axes.iter().map(|x| *x as i64).collect(),
        }
    }
}

impl<'a> From<&'a [i16]> for Axis {
    fn from(axes: &'a [i16]) -> Self {
        Axis {
            axes: axes.iter().map(|x| *x as i64).collect(),
        }
    }
}

impl<'a> From<&'a [i8]> for Axis {
    fn from(axes: &'a [i8]) -> Self {
        Axis {
            axes: axes.iter().map(|x| *x as i64).collect(),
        }
    }
}

impl<'a> From<&'a [i128]> for Axis {
    fn from(axes: &'a [i128]) -> Self {
        Axis {
            axes: axes.iter().map(|x| *x as i64).collect(),
        }
    }
}

impl<'a, const N: usize> From<&'a [i64; N]> for Axis {
    fn from(axes: &'a [i64; N]) -> Self {
        Axis {
            axes: axes.to_vec(),
        }
    }
}

impl<'a, const N: usize> From<&'a [i32; N]> for Axis {
    fn from(axes: &'a [i32; N]) -> Self {
        Axis {
            axes: axes.iter().map(|x| *x as i64).collect(),
        }
    }
}

impl<'a, const N: usize> From<&'a [i16; N]> for Axis {
    fn from(axes: &'a [i16; N]) -> Self {
        Axis {
            axes: axes.iter().map(|x| *x as i64).collect(),
        }
    }
}

impl<'a, const N: usize> From<&'a [i8; N]> for Axis {
    fn from(axes: &'a [i8; N]) -> Self {
        Axis {
            axes: axes.iter().map(|x| *x as i64).collect(),
        }
    }
}

impl<'a, const N: usize> From<&'a [i128; N]> for Axis {
    fn from(axes: &'a [i128; N]) -> Self {
        Axis {
            axes: axes.iter().map(|x| *x as i64).collect(),
        }
    }
}

impl<const N: usize> From<[i64; N]> for Axis {
    fn from(axes: [i64; N]) -> Self {
        Axis {
            axes: axes.to_vec(),
        }
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
            axes: axes.iter().map(|x| *x as i64).collect(),
        }
    }
}

impl From<Vec<usize>> for Axis {
    fn from(axes: Vec<usize>) -> Self {
        Axis {
            axes: axes.iter().map(|x| *x as i64).collect(),
        }
    }
}

impl<'a> From<&'a Vec<usize>> for Axis {
    fn from(axes: &'a Vec<usize>) -> Self {
        Axis {
            axes: axes.iter().map(|x| *x as i64).collect(),
        }
    }
}

impl From<i64> for Axis {
    fn from(axes: i64) -> Self {
        Axis { axes: vec![axes] }
    }
}

impl From<&i64> for Axis {
    fn from(axes: &i64) -> Self {
        Axis { axes: vec![*axes] }
    }
}
