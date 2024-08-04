use crate::tensor_base::_Tensor;

impl<T> _Tensor<T> {
    pub fn pad(
        &self,
        pads: &_Tensor<i64>,
        const_val: Option<T>,
        axes: Option<&[usize]>,
        mode: impl ToString
    ) -> _Tensor<T> {
        todo!()
    }
}
