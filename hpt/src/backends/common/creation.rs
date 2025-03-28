use std::panic::Location;

use hpt_common::error::{base::TensorError, shape::ShapeError};
use hpt_traits::tensor::CommonBounds;
use hpt_types::into_scalar::Cast;

pub(crate) fn geomspace_preprocess_start_step<T>(
    start: f64,
    end: f64,
    n: usize,
    include_end: bool,
) -> std::result::Result<(f64, f64), TensorError>
where
    f64: Cast<T>,
    usize: Cast<T>,
    T: CommonBounds + Cast<f64>,
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
