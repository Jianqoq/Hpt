use std::{ panic::Location, sync::Arc };

use crate::{
    halide::{ store_stmt::StoreStmt, utils::var },
    iter_var::IterVar,
    te::{
        context::Context,
        stages::{ Body, Stage },
        strides_cal_helper::elementwise_strides_cal,
        tensor::{ StridesCal, Tensor },
    },
};

impl Context {
    #[track_caller]
    pub fn concat(&mut self, axis: usize, inputs: &[Tensor]) -> Tensor {
        let id = self.id.borrow().clone();
        *self.id.borrow_mut() += 1;

        let concat_dim = inputs
            .iter()
            .map(|x| x.shape[axis].clone())
            .reduce(|x, y| x + y)
            .unwrap();
        let mut res_shape = inputs[0].shape.as_ref().clone();
        res_shape[axis] = concat_dim;
        let res_shape = Arc::new(res_shape);
        let ret = Tensor {
            shape: res_shape.clone(),
            inputs: Arc::new(
                inputs
                    .iter()
                    .map(|x| x.id)
                    .collect()
            ),
            span: Location::caller(),
            id,
            dtype: inputs[0].dtype,
            strides_cal: Arc::new(move |prev_fn: Vec<StridesCal>| {
                elementwise_strides_cal(prev_fn[0].clone())
            }),
            body_gen: Arc::new(move |inputs: Vec<Body>, is_output: bool, id: usize| {
                let shape = res_shape.clone();
                let mut dims = vec![
                    IterVar::new(0i64, shape[axis].clone(), 1i64, var(format!("ax0")))
                ];
                let mut cnt = 1;
                for (idx, i) in shape.iter().enumerate() {
                    if idx != axis {
                        dims.push(IterVar::new(0i64, i.clone(), 1i64, var(format!("ax{}", cnt))));
                        cnt += 1;
                    }
                }
                if is_output {
                    for i in inputs {
                        if let Body::Stage(stage) = i {
                            let mut stage = stage.clone();

                            assert_eq!(stage.bodys.len(), 1);
                            let body = stage.bodys[0].clone();
                        } else {
                            panic!("input is not a stage");
                        }
                    }
                    let body = Body::Stmt(
                        StoreStmt::make(
                            var(format!("%{}", id)),
                            (0..shape.len()).map(|_| var(format!("ax{}", x))).collect(),
                            axes,
                            steps,
                            strides,
                            val
                        ).into()
                    );
                    let ret = Stage {
                        dims,
                        bodys: todo!(),
                        dtype: todo!(),
                        id,
                        out_id: todo!(),
                        begins: todo!(),
                        steps: todo!(),
                        axes: todo!(),
                    };
                } else {
                }
                todo!()
            }),
        };
        todo!()
    }
}
