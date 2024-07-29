use std::{ panic::Location, sync::Arc };

use tensor_types::dtype::Dtype;

use crate::{
    halide::{ exprs::{ Ge, Gt, Int, Le, Lt, Ne }, prime_expr::PrimeExpr },
    iter_var::IterVar,
    te::{
        bodygen_helper::common_binop,
        context::Context,
        stages::Body,
        strides_cal_helper::binary_strides_cal,
        tensor::{ StridesCal, Tensor },
    },
};

macro_rules! impl_cmp {
    (
        $op_name:ident,
        $($op:tt)*
    ) => {
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
                dtype: Dtype::Bool,
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
                        |_, _| Dtype::Bool,
                        |x, y| $($op)*::make(x, y).into(),
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
    impl_cmp!(gt, Gt);
    impl_cmp!(lt, Lt);
    impl_cmp!(ge, Ge);
    impl_cmp!(le, Le);
    impl_cmp!(eq, crate::halide::exprs::Eq);
    impl_cmp!(ne, Ne);
}
