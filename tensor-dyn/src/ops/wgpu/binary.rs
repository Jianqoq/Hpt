use tensor_traits::{ CommonBounds, NormalBinOps };
use tensor_types::type_promote::NormalOut;

use crate::{ backend::Wgpu, ops::cpu::binary::NormalType, tensor_base::_Tensor };

impl<A, B> NormalBinOps<&_Tensor<B, Wgpu>>
    for _Tensor<A>
    where A: CommonBounds + NormalOut<B>, B: CommonBounds, <A as NormalOut<B>>::Output: CommonBounds
{
    type Output = _Tensor<NormalType<A, B>>;
    type OutputMeta = NormalType<A, B>;
    type InplaceOutput = _Tensor<NormalType<A, B>>;

    fn add_<U>(&self, _: &_Tensor<B, Wgpu>, _: U) -> anyhow::Result<Self::Output>
        where
            U: tensor_traits::TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                tensor_traits::TensorInfo<Self::OutputMeta>
    {
        todo!()
    }

    fn sub_<U>(&self, _: &_Tensor<B, Wgpu>, _: U) -> anyhow::Result<Self::Output>
        where
            U: tensor_traits::TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                tensor_traits::TensorInfo<Self::OutputMeta>
    {
        todo!()
    }

    fn mul_<U>(&self, _: &_Tensor<B, Wgpu>, _: U) -> anyhow::Result<Self::Output>
        where
            U: tensor_traits::TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                tensor_traits::TensorInfo<Self::OutputMeta>
    {
        todo!()
    }

    fn rem_<U>(&self, _: &_Tensor<B, Wgpu>, _: U) -> anyhow::Result<Self::Output>
        where
            U: tensor_traits::TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                tensor_traits::TensorInfo<Self::OutputMeta>
    {
        todo!()
    }

    fn convolve(&self, _: &_Tensor<B, Wgpu>) -> anyhow::Result<Self::Output> {
        todo!()
    }
}
