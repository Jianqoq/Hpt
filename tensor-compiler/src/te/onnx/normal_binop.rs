use std::{ panic::Location, sync::Arc };

use tensor_types::{ dtype::Dtype, type_promote::{ BitWiseOut, FloatOut, NormalOut } };

use crate::{
    halide::{ exprs::Int, prime_expr::PrimeExpr },
    iter_var::IterVar,
    te::{
        bodygen_helper::common_binop,
        context::Context,
        stages::Body,
        strides_cal_helper::binary_strides_cal,
        tensor::{ StridesCal, Tensor },
    },
};

macro_rules! impl_std_binop {
    ($op_name:ident, $ty_infer:ident) => {
        #[track_caller]
        pub fn $op_name(&mut self, a: &Tensor, b: &Tensor) -> Tensor {
            let lhs_shape = a.shape.clone();
            let rhs_shape = b.shape.clone();
            let mut res_shape = vec![];
            let mut lhs_replace = vec![];
            let mut rhs_replace = vec![];
            let mut lhs_new_axes = vec![];
            let mut rhs_new_axes = vec![];
    
            let (lhs_start, rhs_start) = if lhs_shape.len() < rhs_shape.len() {
                (0..rhs_shape.len() - lhs_shape.len()).for_each(|x| {
                    rhs_replace.push((res_shape.len(), x));
                    lhs_new_axes.push(res_shape.len());
                    res_shape.push(rhs_shape[x].clone());
                });
                (0, rhs_shape.len() - lhs_shape.len())
            } else if lhs_shape.len() > rhs_shape.len() {
                (0..lhs_shape.len() - rhs_shape.len()).for_each(|x| {
                    lhs_replace.push((res_shape.len(), x));
                    rhs_new_axes.push(res_shape.len());
                    res_shape.push(lhs_shape[x].clone());
                });
                (lhs_shape.len() - rhs_shape.len(), 0)
            } else {
                (0, 0)
            };
    
            let one = PrimeExpr::Int(Int::make(Dtype::I64, 1));
            lhs_shape[lhs_start..]
                .iter()
                .enumerate()
                .zip(rhs_shape[rhs_start..].iter().enumerate())
                .for_each(|((lhs_idx, x), (rhs_idx, y))| {
                    lhs_replace.push((res_shape.len(), lhs_idx + lhs_start));
                    rhs_replace.push((res_shape.len(), rhs_idx + rhs_start));
                    if x == y {
                        res_shape.push(x.clone());
                    } else if x == &one {
                        res_shape.push(y.clone());
                    } else if y == &one {
                        res_shape.push(x.clone());
                    } else {
                        panic!("Incompatible shapes. {} and {}", x, y);
                    }
                });
            let id = self.id.borrow().clone();
            *self.id.borrow_mut() += 1;
            let res_shape = Arc::new(res_shape);
            let res_shape1 = res_shape.clone();
            let lhs_replace = Arc::new(lhs_replace);
            let rhs_replace = Arc::new(rhs_replace);
            let ret = Tensor {
                shape: res_shape.clone(),
                inputs: Arc::new(vec![a.id, b.id]),
                span: Location::caller(),
                id,
                dtype: a.dtype.$ty_infer(b.dtype),
                strides_cal: Arc::new(move |prev_fn: Vec<StridesCal>| {
                    let lhs_shape = lhs_shape.clone();
                    let rhs_shape = rhs_shape.clone();
                    let res_shape = res_shape1.clone();
                    binary_strides_cal(
                        lhs_shape,
                        rhs_shape,
                        res_shape,
                        prev_fn[0].clone(),
                        prev_fn[1].clone()
                    )
                }),
                body_gen: Arc::new(move |inputs: Vec<Body>, is_output: bool, id: usize| {
                    let dims = res_shape
                        .iter()
                        .enumerate()
                        .map(|(idx, x)| IterVar::new(0i64, x.clone(), 1i64, &format!("ax{}", idx)))
                        .collect::<Vec<IterVar>>();
                    common_binop(
                        is_output,
                        &inputs,
                        &lhs_new_axes,
                        &lhs_replace,
                        &rhs_new_axes,
                        &rhs_replace,
                        |x, y| x.$ty_infer(y),
                        |x, y| x.$ty_infer(y),
                        &dims,
                        id
                    )
                }),
            };
            self.nodes.borrow_mut().insert(id, ret.clone());
            ret
        }
    };
}

impl Context {
    impl_std_binop!(add, _add);
    impl_std_binop!(sub, _sub);
    impl_std_binop!(mul, _mul);
    impl_std_binop!(div, _div);
    impl_std_binop!(rem, _rem);
    impl_std_binop!(bitand, _bitand);
    impl_std_binop!(bitor, _bitor);
    impl_std_binop!(bitxor, _bitxor);
    impl_std_binop!(shl, _shl);
    impl_std_binop!(shr, _shr);
}
