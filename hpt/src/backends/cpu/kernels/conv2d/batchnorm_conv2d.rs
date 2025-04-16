use crate::backend::Cpu;
use crate::backends::cpu::kernels::conv2d::conv2d_new_mp;
use crate::backends::cpu::kernels::matmul::microkernel_trait::MatmulMicroKernel;
use crate::backends::cpu::kernels::normalization::batch_norm::batch_norm;
use crate::tensor_base::_Tensor;
use hpt_allocator::traits::{ Allocator, AllocatorOutputRetrive };
use hpt_common::error::base::TensorError;
use hpt_traits::tensor::CommonBounds;
use hpt_types::dtype::TypeCommon;
use hpt_types::into_scalar::Cast;
use hpt_types::type_promote::FloatOutBinary;
use hpt_types::type_promote::FloatOutUnary;
use hpt_types::type_promote::NormalOutPromote;
use hpt_types::traits::VecTrait;
use super::conv2d;
use super::microkernel_trait::Conv2dMicroKernel;

#[track_caller]
pub(crate) fn batchnorm_conv2d<T, const DEVICE: usize, A>(
    input: &_Tensor<T, Cpu, DEVICE, A>,
    kernels: &_Tensor<T, Cpu, DEVICE, A>,
    mean: &_Tensor<T, Cpu, DEVICE, A>,
    var: &_Tensor<T, Cpu, DEVICE, A>,
    gamma: &_Tensor<T, Cpu, DEVICE, A>,
    beta: &_Tensor<T, Cpu, DEVICE, A>,
    bias: Option<&_Tensor<T, Cpu, DEVICE, A>>,
    eps: T,
    steps: [i64; 2],
    padding: [(i64, i64); 2],
    dilation: [i64; 2],
    post_scalar: Option<fn(T) -> T>,
    post_vec: Option<fn(<T>::Vec) -> <T>::Vec>
)
    -> std::result::Result<_Tensor<T, Cpu, DEVICE, A>, TensorError>
    where
        T: CommonBounds + Conv2dMicroKernel + MatmulMicroKernel,
        T::Vec: FloatOutBinary<Output = T::Vec> + FloatOutUnary<Output = T::Vec>,
        T: FloatOutBinary<Output = T> + FloatOutUnary<Output = T>,
        A: Allocator + Send + Sync,
        A::Output: AllocatorOutputRetrive
{
    let conv_res = if T::STR == "bf16" {
        type F32Vec = <<half::bf16 as NormalOutPromote>::Intermediate as TypeCommon>::Vec;
        type BF16Vec = <half::bf16 as TypeCommon>::Vec;
        let inp = input
            .static_cast::<half::bf16>()
            .expect("static_cast bf16 failed");
        let ker = kernels
            .static_cast::<half::bf16>()
            .expect("static_cast bf16 failed");
        let bias = bias.map(|x| {
            x.static_cast::<half::bf16>()
                .expect("static_cast bf16 failed")
        });
        let res = conv2d_new_mp::conv2d::<half::bf16, DEVICE, A>(
            &inp,
            &ker,
            bias.as_ref(),
            steps,
            padding,
            dilation,
            |x| {
                let vec0 = unsafe { x.read_unaligned() };
                let vec1 = unsafe { x.add(1).read_unaligned() };
                BF16Vec::from_2_f32vec([vec0, vec1])
            },
            |x| unsafe {
                let mut bf16_vec = BF16Vec::splat(half::bf16::from_f32_const(0.0));
                for j in 0..F32Vec::SIZE {
                    bf16_vec[j] = *x.add(j);
                }
                let val_f32 = bf16_vec.high_to_f32vec();
                val_f32
            },
            |x| x.cast(),
            |x| x.cast(),
            None,
            None,
        )?;
        Ok(res.static_cast::<T>()?)
    } else if T::STR == "f16" && !cfg!(target_feature = "neon") {
        type F32Vec = <<half::f16 as NormalOutPromote>::Intermediate as TypeCommon>::Vec;
        type F16Vec = <half::f16 as TypeCommon>::Vec;
        let inp = input
            .static_cast::<half::f16>()
            .expect("static_cast f16 failed");
        let ker = kernels
            .static_cast::<half::f16>()
            .expect("static_cast f16 failed");
        let bias = bias.map(|x| {
            x.static_cast::<half::f16>()
                .expect("static_cast f16 failed")
        });
        let res = conv2d_new_mp::conv2d::<half::f16, DEVICE, A>(
            &inp,
            &ker,
            bias.as_ref(),
            steps,
            padding,
            dilation,
            |x| {
                let vec0 = unsafe { x.read_unaligned() };
                let vec1 = unsafe { x.add(1).read_unaligned() };
                F16Vec::from_2_f32vec([vec0, vec1])
            },
            |x| unsafe {
                let mut f16_vec = F16Vec::splat(half::f16::from_f32_const(0.0));
                for j in 0..F32Vec::SIZE {
                    f16_vec[j] = *x.add(j);
                }
                let val_f32 = f16_vec.high_to_f32vec();
                val_f32
            },
            |x| x.cast(),
            |x| x.cast(),
            None,
            None,
        )?;
        Ok(res.static_cast::<T>()?)
    } else {
        conv2d::conv2d(
            input,
            kernels,
            bias,
            steps,
            padding,
            dilation,
            None,
            None,
        )
    }?;

    batch_norm(
        &conv_res,
        mean,
        var,
        gamma,
        beta,
        eps,
        post_scalar,
        post_vec,
        Some(conv_res.clone())
    )?;

    Ok(conv_res)
}
