use crate::err_handler::ErrHandler;

pub struct Axis {
    pub axes: Vec<i64>,
}

pub fn process_axes<T: Into<Axis>>(axes: T, ndim: usize) -> anyhow::Result<Vec<usize>> {
    let ndim = ndim as i64;
    let axes = axes.into().axes;
    let mut new_axes = Vec::with_capacity(axes.len());
    for i in 0..axes.len() {
        if axes[i] < 0 {
            let val = (axes[i] % ndim) + ndim;
            new_axes.push(val as usize);
        } else {
            if axes[i] >= ndim {
                return Err(
                    ErrHandler::IndexOutOfRange(
                        format!(
                            "Axes {} out of range(Should be {}..{}). Pos: {}",
                            axes[i],
                            0,
                            ndim,
                            i
                        )
                    ).into()
                );
            }
            new_axes.push(axes[i] as usize);
        }
    }
    Ok(new_axes)
}

impl<'a> From<&'a [i64]> for Axis {
    fn from(axes: &'a [i64]) -> Self {
        Axis { axes: axes.to_vec() }
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
