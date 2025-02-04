use std::panic::Location;

use tensor_common::error::{base::TensorError, shape::ShapeError};
use tensor_traits::CommonBounds;
use tensor_types::into_scalar::IntoScalar;

pub(crate) fn geomspace_preprocess_start_step<T>(
    start: f64,
    end: f64,
    n: usize,
    include_end: bool,
) -> std::result::Result<(f64, f64), TensorError>
where
    f64: IntoScalar<T>,
    usize: IntoScalar<T>,
    T: CommonBounds + IntoScalar<f64>,
{
    let float_n = n as f64;
    let step = if include_end {
        if start >= 0.0 && end > 0.0 {
            (end.log10() - start.log10()) / (float_n - 1.0)
        } else if start < 0.0 && end < 0.0 {
            (end.abs().log10() - start.abs().log10()) / (float_n - 1.0)
        } else {
            return Err(ShapeError::GeomSpaceError {
                start,
                end,
                location: Location::caller(),
            }
            .into());
        }
    } else if start >= 0.0 && end > 0.0 {
        (end.log10() - start.log10()) / float_n
    } else if start < 0.0 && end < 0.0 {
        (end.abs().log10() - start.abs().log10()) / float_n
    } else {
        return Err(ShapeError::GeomSpaceError {
            start,
            end,
            location: Location::caller(),
        }
        .into());
    };
    let start = if start > 0.0 {
        start.log10()
    } else {
        start.abs().log10()
    };
    Ok((start, step))
}
