use std::{panic::Location, sync::Arc};

use tensor_common::{
    axis::{process_axes, Axis},
    err_handler::ErrHandler,
    shape::Shape,
    shape_utils::{get_broadcast_axes_from, mt_intervals, try_pad_shape},
    strides::Strides,
};

use crate::iterator_traits::{ParStridedHelper, StridedHelper};

#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn par_reshape<S: Into<Shape>, T: ParStridedHelper>(mut iterator: T, shape: S) -> T {
    let tmp = shape.into();
    let res_shape = tmp;
    if iterator._layout().shape() == &res_shape {
        return iterator;
    }
    let size = res_shape.size() as usize;
    let inner_loop_size = res_shape[res_shape.len() - 1] as usize;
    let outer_loop_size = size / inner_loop_size;
    let num_threads;
    if outer_loop_size < rayon::current_num_threads() {
        num_threads = outer_loop_size;
    } else {
        num_threads = rayon::current_num_threads();
    }
    let intervals = mt_intervals(outer_loop_size, num_threads);
    let len = intervals.len();
    iterator._set_intervals(Arc::new(intervals));
    iterator._set_end_index(len);
    let self_size = iterator._layout().size();

    if size > (self_size as usize) {
        let self_shape = try_pad_shape(iterator._layout().shape(), res_shape.len());

        let axes_to_broadcast =
            get_broadcast_axes_from(&self_shape, &res_shape, Location::caller())
                .expect("Cannot broadcast shapes");

        let mut new_strides = vec![0; res_shape.len()];
        new_strides
            .iter_mut()
            .rev()
            .zip(iterator._layout().strides().iter().rev())
            .for_each(|(a, b)| {
                *a = *b;
            });
        for &axis in axes_to_broadcast.iter() {
            assert_eq!(self_shape[axis], 1);
            new_strides[axis] = 0;
        }
        iterator._set_last_strides(new_strides[new_strides.len() - 1]);
        iterator._set_strides(new_strides.into());
    } else {
        ErrHandler::check_size_match(iterator._layout().shape().size(), res_shape.size()).unwrap();
        if let Some(new_strides) = iterator._layout().is_reshape_possible(&res_shape) {
            iterator._set_strides(new_strides);
            iterator._set_last_strides(
                iterator._layout().strides()[iterator._layout().strides().len() - 1],
            );
        } else {
            ErrHandler::IterInplaceReshapeError(
                iterator._layout().shape().clone(),
                res_shape.clone(),
                iterator._layout().strides().clone(),
                Location::caller(),
            );
        }
    }

    iterator._set_shape(res_shape.clone());
    iterator
}

#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn par_transpose<AXIS: Into<Axis>, T: ParStridedHelper>(
    mut iterator: T,
    axes: AXIS,
) -> T {
    let axes = process_axes(axes, iterator._layout().shape().len()).unwrap();

    let mut new_shape = iterator._layout().shape().to_vec();
    for i in axes.iter() {
        new_shape[*i] = iterator._layout().shape()[axes[*i]];
    }
    let mut new_strides = iterator._layout().strides().to_vec();
    for i in axes.iter() {
        new_strides[*i] = iterator._layout().strides()[axes[*i]];
    }
    let new_strides: Strides = new_strides.into();
    let new_shape = Arc::new(new_shape);
    let outer_loop_size =
        (new_shape.iter().product::<i64>() as usize) / (new_shape[new_shape.len() - 1] as usize);
    let num_threads;
    if outer_loop_size < rayon::current_num_threads() {
        num_threads = outer_loop_size;
    } else {
        num_threads = rayon::current_num_threads();
    }
    let intervals = Arc::new(mt_intervals(outer_loop_size, num_threads));
    let len = intervals.len();
    iterator._set_intervals(intervals.clone());
    iterator._set_end_index(len);

    iterator._set_last_strides(new_strides[new_strides.len() - 1]);
    iterator._set_strides(new_strides);
    iterator._set_shape(Shape::from(new_shape));
    iterator
}

#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn par_expand<S: Into<Shape>, T: ParStridedHelper>(mut iterator: T, shape: S) -> T {
    let res_shape = shape.into();

    let new_strides = iterator._layout().expand_strides(&res_shape);

    let outer_loop_size =
        (res_shape.iter().product::<i64>() as usize) / (res_shape[res_shape.len() - 1] as usize);
    let num_threads;
    if outer_loop_size < rayon::current_num_threads() {
        num_threads = outer_loop_size;
    } else {
        num_threads = rayon::current_num_threads();
    }
    let intervals = Arc::new(mt_intervals(outer_loop_size, num_threads));
    let len = intervals.len();
    iterator._set_intervals(intervals.clone());
    iterator._set_end_index(len);
    iterator._set_shape(res_shape.clone());
    iterator._set_strides(new_strides);
    iterator
}

#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn reshape<S: Into<Shape>, T: StridedHelper>(mut iterator: T, shape: S) -> T {
    let tmp = shape.into();
    let res_shape = tmp;
    if iterator._layout().shape() == &res_shape {
        return iterator;
    }
    let size = res_shape.size() as usize;
    let self_size = iterator._layout().size();

    if size > (self_size as usize) {
        let self_shape = try_pad_shape(iterator._layout().shape(), res_shape.len());

        let axes_to_broadcast =
            get_broadcast_axes_from(&self_shape, &res_shape, Location::caller())
                .expect("Cannot broadcast shapes");

        let mut new_strides = vec![0; res_shape.len()];
        new_strides
            .iter_mut()
            .rev()
            .zip(iterator._layout().strides().iter().rev())
            .for_each(|(a, b)| {
                *a = *b;
            });
        for &axis in axes_to_broadcast.iter() {
            assert_eq!(self_shape[axis], 1);
            new_strides[axis] = 0;
        }
        iterator._set_last_strides(new_strides[new_strides.len() - 1]);
        iterator._set_strides(new_strides.into());
    } else {
        ErrHandler::check_size_match(
            iterator._layout().shape().inner().iter().product(),
            res_shape.size(),
        )
        .unwrap();
        if let Some(new_strides) = iterator._layout().is_reshape_possible(&res_shape) {
            iterator._set_strides(new_strides);
            iterator._set_last_strides(
                iterator._layout().strides()[iterator._layout().strides().len() - 1],
            );
        } else {
            let error = ErrHandler::IterInplaceReshapeError(
                iterator._layout().shape().clone(),
                res_shape.clone(),
                iterator._layout().strides().clone(),
                Location::caller(),
            );
            panic!("{}", error);
        }
    }

    iterator._set_shape(res_shape.clone());
    iterator
}

#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn expand<T: StridedHelper, S: Into<Shape>>(mut iterator: T, shape: S) -> T {
    let res_shape: Shape = shape.into();
    let new_strides = iterator._layout().expand_strides(&res_shape);
    iterator._set_shape(res_shape.clone());
    iterator._set_strides(new_strides);
    iterator
}

#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn transpose<T: StridedHelper, AXIS: Into<Axis>>(mut iterator: T, axes: AXIS) -> T {
    // ErrHandler::check_axes_in_range(self.shape().len(), axes).unwrap();
    let axes = process_axes(axes, iterator._layout().shape().len()).unwrap();

    let mut new_shape = iterator._layout().shape().to_vec();
    for i in axes.iter() {
        new_shape[*i] = iterator._layout().shape()[axes[*i]];
    }
    let mut new_strides = iterator._layout().strides().to_vec();
    for i in axes.iter() {
        new_strides[*i] = iterator._layout().strides()[axes[*i]];
    }
    let new_strides: Strides = new_strides.into();
    let new_shape = Arc::new(new_shape);

    iterator._set_last_strides(new_strides[new_strides.len() - 1]);
    iterator._set_strides(new_strides);
    iterator._set_shape(Shape::from(new_shape));
    iterator
}
