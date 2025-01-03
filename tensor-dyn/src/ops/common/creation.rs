use std::panic::Location;

use tensor_common::err_handler::TensorError;
use tensor_traits::CommonBounds;
use tensor_types::{convertion::Convertor, into_scalar::IntoScalar};

pub(crate) fn geomspace_preprocess_start_step<T>(
    start: f64,
    end: f64,
    n: usize,
    include_end: bool,
) -> std::result::Result<(f64, f64), TensorError>
where
    f64: IntoScalar<T>,
    usize: IntoScalar<T>,
    T: CommonBounds + Convertor,
{
    let float_n = n.to_f64();
    let step = if include_end {
        if start >= 0.0 && end > 0.0 {
            (end.log10() - start.log10()) / (float_n - 1.0)
        } else if start < 0.0 && end < 0.0 {
            (end.abs().log10() - start.abs().log10()) / (float_n - 1.0)
        } else {
            return Err(TensorError::GeomSpaceStartEndError(
                start,
                end,
                Location::caller(),
            ));
        }
    } else if start >= 0.0 && end > 0.0 {
        (end.log10() - start.log10()) / float_n
    } else if start < 0.0 && end < 0.0 {
        (end.abs().log10() - start.abs().log10()) / float_n
    } else {
        return Err(TensorError::GeomSpaceStartEndError(
            start,
            end,
            Location::caller(),
        ));
    };
    let start = if start > 0.0 {
        start.log10()
    } else {
        start.abs().log10()
    };
    Ok((start, step))
}
