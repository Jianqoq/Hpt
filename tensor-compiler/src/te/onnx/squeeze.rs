use std::{ collections::HashMap, panic::Location, sync::Arc };

use crate::{
    halide::prime_expr::PrimeExpr,
    te::{ context::Context, hstrides::HStrides, stages::Body, tensor::{ StridesCal, Tensor } },
};

impl Context {
    #[track_caller]
    pub fn squeeze(&mut self, a: &Tensor, axes: &[i64]) -> Tensor {
        let id = self.id.borrow().clone();
        *self.id.borrow_mut() += 1;
        let mut axes = axes.to_vec();
        let one: PrimeExpr = (1i64).into();
        for i in 0..axes.len() {
            if axes[i] < 0 {
                axes[i] += a.shape.len() as i64;
            }
            if &a.shape[axes[i] as usize] != &one {
                panic!(
                    "cannot select an axis to squeeze out which has size not equal to one, try to squeeze axis {} with size {} in shape {:?} at {:?}",
                    axes[i],
                    a.shape[axes[i] as usize],
                    a.shape,
                    Location::caller()
                );
            }
        }
        let new_shape: Arc<Vec<PrimeExpr>> = Arc::new(
            a.shape
                .iter()
                .enumerate()
                .filter(|&(i, _)| !axes.contains(&(i as i64)))
                .map(|(_, x)| x.clone())
                .collect()
        );
        let dtype = a.dtype.clone();
        let ret = Tensor {
            shape: new_shape.clone(),
            inputs: Arc::new(vec![a.id]),
            span: Location::caller(),
            id,
            dtype,
            strides_cal: Arc::new(move |prev_fn: Vec<StridesCal>| {
                {
                    let prev_func = prev_fn[0].clone();
                    let axes = axes.clone();
                    Arc::new(move |map: &HashMap<Arc<String>, i64>| {
                        let strides = prev_func(map);
                        let squeezed_strides = strides
                            .iter()
                            .map(|strides| {
                                let mut new_strides = vec![];
                                for (i, &stride) in strides.strides.iter().enumerate() {
                                    if !axes.contains(&(i as i64)) {
                                        new_strides.push(stride);
                                    }
                                }
                                HStrides {
                                    strides: new_strides,
                                    reduced_dim: strides.reduced_dim,
                                    offset: strides.offset,
                                }
                            })
                            .collect::<Vec<HStrides>>();
                        squeezed_strides
                    })
                }
            }),
            body_gen: Arc::new(move |inputs: Vec<Body>, is_output: bool, id: usize| {
                let new_shape = new_shape.clone();
                if is_output {
                    if let Body::Stage(stage) = &inputs[0] {
                        
                    } else {
                        panic!("squeeze body_gen should have a stage input");
                    }
                } else {
                    if let Body::Stage(stage) = &inputs[0] {
                    } else {
                        panic!("squeeze body_gen should have a stage input");
                    }
                }
                todo!()
            }),
        };
        self.nodes.borrow_mut().insert(id, ret.clone());
        ret
    }
}
