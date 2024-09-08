use tensor_traits::CommonBounds;
use tensor_types::type_promote::NormalOut;
use crate::{ backend::Wgpu, ops::wgpu::binary_normal::binop, tensor_base::_Tensor };

macro_rules! impl_std_ops {
    ($trait:ident, $op_name:ident, $op_str:expr) => {
        impl<T, U> std::ops::$trait<_Tensor<U, Wgpu>>
            for _Tensor<T, Wgpu>
            where
                T: CommonBounds + NormalOut<U> + bytemuck::Pod,
                U: CommonBounds + bytemuck::Pod,
                <T as NormalOut<U>>::Output: CommonBounds + bytemuck::Pod
                {
                    type Output = _Tensor<<T as NormalOut<U>>::Output, Wgpu>;
                    #[track_caller]
                    fn $op_name(self, rhs: _Tensor<U, Wgpu>) -> Self::Output {
                        binop($op_str, &self, &rhs)
                    }
                }

        impl<T, U> std::ops::$trait<&_Tensor<U, Wgpu>>
            for _Tensor<T, Wgpu>
            where
                T: CommonBounds + NormalOut<U> + bytemuck::Pod,
                U: CommonBounds + bytemuck::Pod,
                <T as NormalOut<U>>::Output: CommonBounds + bytemuck::Pod
                {
                    type Output = _Tensor<<T as NormalOut<U>>::Output, Wgpu>;
                    #[track_caller]
                    fn $op_name(self, rhs: &_Tensor<U, Wgpu>) -> Self::Output {
                        binop($op_str, &self, &rhs)
                    }
                }

        impl<T, U> std::ops::$trait<_Tensor<U, Wgpu>>
            for &_Tensor<T, Wgpu>
            where
                T: CommonBounds + NormalOut<U> + bytemuck::Pod,
                U: CommonBounds + bytemuck::Pod,
                <T as NormalOut<U>>::Output: CommonBounds + bytemuck::Pod
                {
                    type Output = _Tensor<<T as NormalOut<U>>::Output, Wgpu>;
                    #[track_caller]
                    fn $op_name(self, rhs: _Tensor<U, Wgpu>) -> Self::Output {
                        binop($op_str, &self, &rhs)
                    }
                }

        impl<T, U> std::ops::$trait<&_Tensor<U, Wgpu>>
            for &_Tensor<T, Wgpu>
            where
                T: CommonBounds + NormalOut<U> + bytemuck::Pod,
                U: CommonBounds + bytemuck::Pod,
                <T as NormalOut<U>>::Output: CommonBounds + bytemuck::Pod
                {
                    type Output = _Tensor<<T as NormalOut<U>>::Output, Wgpu>;
                    #[track_caller]
                    fn $op_name(self, rhs: &_Tensor<U, Wgpu>) -> Self::Output {
                        binop($op_str, &self, &rhs)
                    }
                }
    };
}

impl_std_ops!(Add, add, "+");
impl_std_ops!(Sub, sub, "-");
impl_std_ops!(Mul, mul, "*");
impl_std_ops!(Div, div, "/");
impl_std_ops!(Rem, rem, "%");
impl_std_ops!(BitAnd, bitand, "&");
impl_std_ops!(BitOr, bitor, "|");
impl_std_ops!(BitXor, bitxor, "^");
impl_std_ops!(Shl, shl, "<<");
impl_std_ops!(Shr, shr, ">>");
