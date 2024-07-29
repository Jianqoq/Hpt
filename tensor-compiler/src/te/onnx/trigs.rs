use std::{ panic::Location, sync::Arc };

use tensor_types::type_promote::FloatOut;

use crate::te::{
    bodygen_helper::common_uaryop,
    context::Context,
    stages::Body,
    strides_cal_helper::elementwise_strides_cal,
    tensor::{ StridesCal, Tensor },
};

macro_rules! impl_trig {
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
                dtype: a.dtype.clone(),
                strides_cal: Arc::new(move |prev_fn: Vec<StridesCal>| {
                    elementwise_strides_cal(prev_fn[0].clone())
                }),
                body_gen: Arc::new(move |inputs: Vec<Body>, is_output: bool, id: usize| {
                    common_uaryop(is_output, &inputs, &shape, |x| x.$infer_name(), |x| x.$infer_name(), id)
                }),
            };
            self.nodes.borrow_mut().insert(id, ret.clone());
            ret
        }
    };
}

impl Context {
    impl_trig!(sin, _sin);
    impl_trig!(cos, _cos);
    impl_trig!(tan, _tan);
    impl_trig!(asin, _asin);
    impl_trig!(acos, _acos);
    impl_trig!(atan, _atan);
    impl_trig!(sinh, _sinh);
    impl_trig!(cosh, _cosh);
    impl_trig!(tanh, _tanh);
    impl_trig!(asinh, _asinh);
    impl_trig!(acosh, _acosh);
    impl_trig!(atanh, _atanh);
}
