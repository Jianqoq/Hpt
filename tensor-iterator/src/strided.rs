use std::{ fmt::Display, panic::Location, sync::Arc };
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
use tensor_traits::tensor::{ CommonBounds, TensorInfo };
use tensor_common::shape_utils::predict_broadcast_shape;
use crate::{ iterator_traits::{ IterGetSet, ShapeManipulator }, strided_zip::StridedZip };

#[derive(Clone)]
pub struct Strided<T> {
    pub(crate) ptr: Pointer<T>,
    pub(crate) layout: Layout,
    pub(crate) prg: Vec<i64>,
    pub(crate) last_stride: i64,
}

impl<T: CommonBounds> Strided<T> {
    pub fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    pub fn strides(&self) -> &Strides {
        self.layout.strides()
    }

    pub fn new<U: TensorInfo<T>>(tensor: U) -> Self {
        Strided {
            ptr: tensor.ptr(),
            layout: tensor.layout().clone(),
            prg: vec![],
            last_stride: tensor.strides()[tensor.strides().len() - 1],
        }
    }

    #[track_caller]
    pub fn zip<'a, C>(mut self, mut other: C) -> StridedZip<'a, Self, C>
        where C: 'a + IterGetSet, <C as IterGetSet>::Item: Send
    {
        let new_shape = predict_broadcast_shape(
            self.shape(),
            other.shape(),
            Location::caller()
        ).expect("Cannot broadcast shapes");

        other.broadcast_set_strides(&new_shape);
        self.broadcast_set_strides(&new_shape);

        other.set_shape(new_shape.clone());
        self.set_shape(new_shape.clone());

        StridedZip::new(self, other)
    }
}

impl<T: Copy + Display> IterGetSet for Strided<T> {
    type Item = T;

    fn set_end_index(&mut self, _: usize) {
        panic!("single thread iterator does not support set_end_index");
    }

    fn set_intervals(&mut self, _: Arc<Vec<(usize, usize)>>) {
        panic!("single thread iterator does not support set_intervals");
    }

    fn set_strides(&mut self, strides: Strides) {
        self.layout.set_strides(strides);
    }

    fn set_shape(&mut self, shape: Shape) {
        self.layout.set_shape(shape);
    }

    fn intervals(&self) -> &Arc<Vec<(usize, usize)>> {
        panic!("single thread iterator does not support intervals");
    }

    fn strides(&self) -> &Strides {
        self.layout.strides()
    }

    fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    fn broadcast_set_strides(&mut self, shape: &Shape) {
        let self_shape = try_pad_shape(self.shape(), shape.len());
        self.set_strides(preprocess_strides(&self_shape, self.strides()).into());
        self.last_stride = self.strides()[self.strides().len() - 1];
    }

    fn outer_loop_size(&self) -> usize {
        (self.shape().iter().product::<i64>() as usize) / self.inner_loop_size()
    }

    fn inner_loop_size(&self) -> usize {
        self.shape().last().unwrap().clone() as usize
    }

    fn next(&mut self) {
        for j in (0..(self.shape().len() as i64) - 1).rev() {
            let j = j as usize;
            if self.prg[j] < self.shape()[j] - 1 {
                self.prg[j] += 1;
                self.ptr.offset(self.strides()[j]);
                break;
            } else {
                self.prg[j] = 0;
                self.ptr.offset(-self.strides()[j] * (self.shape()[j] - 1));
            }
        }
    }

    fn inner_loop_next(&mut self, index: usize) -> Self::Item {
        unsafe { *self.ptr.get_ptr().add(index * (self.last_stride as usize)) }
    }

    fn set_prg(&mut self, prg: Vec<i64>) {
        self.prg = prg;
    }
}

impl<T: Copy + Display> ShapeManipulator for Strided<T> {
    #[track_caller]
    fn reshape<S: Into<Shape>>(mut self, shape: S) -> Self {
        let tmp = shape.into();
        let res_shape = tmp;
        if self.layout.shape() == &res_shape {
            return self;
        }
        let size = res_shape.size() as usize;
        let self_size = self.layout.size();

        if size > (self_size as usize) {
            let self_shape = try_pad_shape(self.shape(), res_shape.len());

            let axes_to_broadcast = get_broadcast_axes_from(
                &self_shape,
                &res_shape,
                Location::caller()
            ).expect("Cannot broadcast shapes");

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
            ErrHandler::check_size_match(self.layout.shape().size(), res_shape.size()).unwrap();
            if let Some(new_strides) = self.layout.is_reshape_possible(&res_shape) {
                self.set_strides(new_strides);
                self.last_stride = self.strides()[self.strides().len() - 1];
            } else {
                let error = ErrHandler::IterInplaceReshapeError(
                    self.shape().clone(),
                    res_shape.clone(),
                    self.strides().clone(),
                    Location::caller()
                );
                panic!("{}", error);
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
        let res_shape = shape.into();

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
