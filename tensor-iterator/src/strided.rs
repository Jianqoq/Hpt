use std::sync::Arc;

use rayon::iter::plumbing::{ Folder, UnindexedProducer };
use tensor_common::{
    axis::{ process_axes, Axis },
    err_handler::ErrHandler,
    layout::Layout,
    pointer::Pointer,
    shape::Shape,
    shape_utils::{ get_broadcast_axes_from, mt_intervals, try_pad_shape },
    strides::Strides,
    strides_utils::preprocess_strides,
};
use tensor_traits::tensor::TensorInfo;

use crate::iterator_traits::{ IterGetSet, ShapeManipulator };

#[derive(Clone)]
pub struct Strided<T> {
    pub(crate) ptr: Pointer<T>,
    pub(crate) layout: Layout,
    pub(crate) prg: Vec<i64>,
    pub(crate) intervals: Arc<Vec<(usize, usize)>>,
    pub(crate) start_index: usize,
    pub(crate) end_index: usize,
    pub(crate) last_stride: i64,
}

impl<T> Strided<T> {
    pub fn shape(&self) -> &Shape {
        &self.layout.shape()
    }

    pub fn strides(&self) -> &Strides {
        &self.layout.strides()
    }

    pub fn new<U: TensorInfo<T>>(tensor: U) -> Self {
        let inner_loop_size = tensor.shape()[tensor.shape().len() - 1] as usize;
        let outer_loop_size = tensor.size() / (inner_loop_size as usize);
        let num_threads;
        if outer_loop_size < rayon::current_num_threads() {
            num_threads = outer_loop_size;
        } else {
            num_threads = rayon::current_num_threads();
        }
        let intervals = mt_intervals(outer_loop_size, num_threads);
        let len = intervals.len();
        Strided {
            ptr: tensor.ptr(),
            layout: tensor.layout().clone(),
            prg: vec![],
            intervals: Arc::new(intervals),
            start_index: 0,
            end_index: len,
            last_stride: tensor.strides()[tensor.strides().len() - 1],
        }
    }
}

impl<T> IterGetSet for Strided<T> {
    type Item = T;

    fn set_end_index(&mut self, end_index: usize) {
        self.end_index = end_index;
    }

    fn set_intervals(&mut self, intervals: Arc<Vec<(usize, usize)>>) {
        self.intervals = intervals;
    }

    fn set_strides(&mut self, last_stride: Strides) {
        self.last_stride = last_stride[0];
    }

    fn set_shape(&mut self, shape: Shape) {
        self.layout.set_shape(shape);
    }

    fn intervals(&self) -> &Arc<Vec<(usize, usize)>> {
        &self.intervals
    }

    fn strides(&self) -> &Strides {
        &self.layout.strides()
    }

    fn shape(&self) -> &Shape {
        &self.layout.shape()
    }

    fn broadcast_set_strides(&mut self, shape: &Shape) {
        let self_shape = try_pad_shape(&self.shape(), shape.len());
        self.set_strides(preprocess_strides(&self_shape, &self.strides()).into());
        self.last_stride = self.strides()[self.strides().len() - 1];
    }
}

impl<T> UnindexedProducer for Strided<T> where T: Clone + Sync + Copy {
    type Item = T;

    fn split(mut self) -> (Self, Option<Self>) {
        if self.end_index - self.start_index <= 1 {
            let mut curent_shape_prg: Vec<i64> = vec![0; self.shape().len()];
            let mut amount =
                self.intervals[self.start_index].0 * (*self.shape().last().unwrap() as usize);
            let mut index = 0;
            for j in (0..=self.shape().len() - 1).rev() {
                curent_shape_prg[j] = (amount as i64) % self.shape()[j];
                amount /= self.shape()[j] as usize;
                index += curent_shape_prg[j] * self.strides()[j];
            }
            self.ptr.offset(index);
            self.prg = curent_shape_prg;
            let mut new_shape = self.shape().to_vec();
            new_shape.iter_mut().for_each(|x| {
                *x -= 1;
            });
            self.last_stride = self.strides()[self.strides().len() - 1];
            self.set_shape(Shape::from(new_shape));
            return (self, None);
        }
        let _left_interval = &self.intervals[self.start_index..self.end_index];
        let left = _left_interval.len() / 2;
        let right = _left_interval.len() / 2 + (_left_interval.len() % 2);
        (
            Strided {
                ptr: self.ptr,
                layout: self.layout.clone(),
                prg: vec![],
                intervals: self.intervals.clone(),
                start_index: self.start_index,
                end_index: self.start_index + left,
                last_stride: self.last_stride,
            },
            Some(Strided {
                ptr: self.ptr,
                layout: self.layout.clone(),
                prg: vec![],
                intervals: self.intervals.clone(),
                start_index: self.start_index + left,
                end_index: self.start_index + left + right,
                last_stride: self.last_stride,
            }),
        )
    }

    fn fold_with<F>(self, folder: F) -> F where F: Folder<Self::Item> {
        folder
    }
}

impl<T> ShapeManipulator for Strided<T> {
    fn reshape<S: Into<Shape>>(mut self, shape: S) -> Self {
        let tmp = shape.into();
        let res_shape = Shape::from(tmp);
        if self.layout.shape() == &res_shape {
            return self;
        }
        let size = res_shape.size() as usize;
        let inner_loop_size = res_shape[res_shape.len() - 1] as usize;
        let outer_loop_size = size / (inner_loop_size as usize);
        let num_threads;
        if outer_loop_size < rayon::current_num_threads() {
            num_threads = outer_loop_size;
        } else {
            num_threads = rayon::current_num_threads();
        }
        let intervals = mt_intervals(outer_loop_size, num_threads);
        let len = intervals.len();
        self.set_intervals(Arc::new(intervals));
        self.set_end_index(len);
        let self_size = self.layout.size();

        if size > self_size {
            let self_shape = try_pad_shape(&self.shape(), res_shape.len());

            let axes_to_broadcast = get_broadcast_axes_from(&self_shape, &res_shape).expect(
                "Cannot broadcast shapes"
            );

            let mut new_strides = vec![0; res_shape.len()];
            new_strides
                .iter_mut()
                .rev()
                .zip(self.strides().iter().rev())
                .for_each(|(a, b)| {
                    *a = *b;
                });
            for &axis in axes_to_broadcast.iter() {
                assert_eq!(self_shape[axis], 1);
                new_strides[axis] = 0;
            }
            self.last_stride = new_strides[new_strides.len() - 1];
            self.set_strides(new_strides.into());
        } else {
            ErrHandler::check_size_match(self.layout.shape(), &res_shape).unwrap();
            if let Some(new_strides) = self.layout.is_reshape_possible(&res_shape) {
                self.set_strides(new_strides);
                self.last_stride = self.strides()[self.strides().len() - 1];
            } else {
                ErrHandler::raise_requires_allocation_when_use_iterator(
                    self.shape(),
                    self.strides()
                ).unwrap();
            }
        }

        self.set_shape(res_shape.clone());
        self
    }

    fn transpose<AXIS: Into<Axis>>(mut self, axes: AXIS) -> Self {
        // ErrHandler::check_axes_in_range(self.shape().len(), axes).unwrap();
        let axes = process_axes(axes, self.shape().len()).unwrap();

        let mut new_shape = self.shape().to_vec();
        for i in axes.iter() {
            new_shape[*i] = self.shape()[axes[*i]];
        }
        let mut new_strides = self.strides().to_vec();
        for i in axes.iter() {
            new_strides[*i] = self.strides()[axes[*i]];
        }
        let new_strides: Strides = new_strides.into();
        let new_shape = Arc::new(new_shape);
        let outer_loop_size =
            (new_shape.iter().product::<i64>() as usize) /
            (new_shape[new_shape.len() - 1] as usize);
        let num_threads;
        if outer_loop_size < rayon::current_num_threads() {
            num_threads = outer_loop_size;
        } else {
            num_threads = rayon::current_num_threads();
        }
        let intervals = Arc::new(mt_intervals(outer_loop_size, num_threads));
        let len = intervals.len();
        self.set_intervals(intervals.clone());
        self.set_end_index(len);

        self.last_stride = new_strides[new_strides.len() - 1];
        self.set_strides(new_strides);
        self.set_shape(Shape::from(new_shape));
        self
    }

    fn expand<S: Into<Shape>>(mut self, shape: S) -> Self {
        let res_shape = Shape::from(shape.into());

        ErrHandler::check_expand_dims(self.shape(), &res_shape).unwrap();

        let new_strides = self.layout.expand_strides(&res_shape);

        let outer_loop_size =
            (res_shape.iter().product::<i64>() as usize) /
            (res_shape[res_shape.len() - 1] as usize);
        let num_threads;
        if outer_loop_size < rayon::current_num_threads() {
            num_threads = outer_loop_size;
        } else {
            num_threads = rayon::current_num_threads();
        }
        let intervals = Arc::new(mt_intervals(outer_loop_size, num_threads));
        let len = intervals.len();
        self.set_intervals(intervals.clone());
        self.set_end_index(len);
        self.set_shape(res_shape.clone());
        self.set_strides(new_strides);
        self
    }
}
