use std::{ panic::Location, sync::Arc };

use tensor_types::type_promote::{ Eval, FloatOut, NormalOut };

use crate::te::{
    bodygen_helper::common_uaryop,
    context::Context,
    stages::Body,
    strides_cal_helper::elementwise_strides_cal,
    tensor::{ StridesCal, Tensor },
};

macro_rules! impl_normal_uary {
    ($op_name:ident, $infer_name:ident) => {
        #[track_caller]
        pub fn $op_name(&mut self, a: &Tensor) -> Tensor {
            let id = self.id.borrow().clone();
            *self.id.borrow_mut() += 1;
            let shape = a.shape.clone();
            let ret = Tensor {
                shape: a.shape.clone(),
                inputs: Arc::new(vec![a.id]),
                span: Location::caller(),
                id,
                dtype: a.dtype.$infer_name(),
                strides_cal: Arc::new(move |prev_fn: Vec<StridesCal>| {
                    elementwise_strides_cal(prev_fn[0].clone())
                }),
                body_gen: Arc::new(move |inputs: Vec<Body>, is_output: bool, id: usize| {
                    common_uaryop(is_output, &inputs, &shape, |x| x.$infer_name(), stringify!($op_name), id)
                }),
            };
            self.nodes.borrow_mut().insert(id, ret.clone());
            ret
        }
    };
}

impl Context {
    impl_normal_uary!(exp, _exp);
    impl_normal_uary!(log10, _log10);
    impl_normal_uary!(log2, _log2);
    impl_normal_uary!(ln, _ln);
    impl_normal_uary!(sqrt, _sqrt);
    impl_normal_uary!(square, _square);
    impl_normal_uary!(abs, _abs);
    impl_normal_uary!(exp2, _exp2);
    impl_normal_uary!(isnan, _is_nan);
    impl_normal_uary!(istrue, _is_true);
    impl_normal_uary!(isinf, _is_inf);
    impl_normal_uary!(ceil, _ceil);
    impl_normal_uary!(floor, _floor);
    impl_normal_uary!(erf, _erf);
}
