use std::sync::Arc;

use crate::ops::cuda::cuda_utils::{
    compile_kernel, get_array_str, get_include_1, get_module_name_1,
};
use crate::{tensor_base::_Tensor, Cuda, Tensor};
use crate::{Backend, ALIGN};
use cudarc::driver::{CudaDevice, DeviceRepr, LaunchAsync};
use tensor_allocator::CUDA_CACHE;
use tensor_common::{layout::Layout, pointer::Pointer, shape::Shape};
use tensor_traits::TensorCreator;
use tensor_traits::{CommonBounds, TensorAlloc, TensorInfo, TensorLike};
use tensor_types::convertion::Convertor;
use tensor_types::into_scalar::IntoScalar;

use super::cuda_utils::compute_kernel_launch_config;
use super::unary::uary_fn_with_out_simd;

impl<T, const DEVICE_ID: usize> TensorLike<T> for _Tensor<T, Cuda, DEVICE_ID>
where
    T: CommonBounds + DeviceRepr,
{
    fn as_raw(&self) -> &[T] {
        unimplemented!()
    }

    fn as_raw_mut(&mut self) -> &mut [T] {
        unimplemented!()
    }

    fn contiguous(&self) -> anyhow::Result<Self> {
        let res = Self::empty(self.shape().clone())?;
        let shape_str = get_array_str(self.shape());
        let strides_str = get_array_str(self.strides());
        let include = get_include_1::<T>();
        let module_name = get_module_name_1("ctg", self);
        let map = compile_kernel(
            &module_name,
            &format!(
                "
                    {include}
                    __constant__ long long shape[] = {{{}}};
                    __constant__ long long strides[] = {{{}}};
                    extern \"C\" __global__ void contiguous({} *out, {} *inp)
                    {{
                        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                        size_t stride = blockDim.x * gridDim.x;
                        while (idx < {})
                        {{
                            long amount = idx;
                            long offset = 0;
                            #pragma unroll
                            for (int j = {} - 1; j >= 0; j--)
                            {{
                                offset += amount % shape[j] * strides[j];
                                amount /= shape[j];
                            }}
                            out[idx] = inp[offset];
                            idx += stride;
                        }}
                    }}",
                shape_str,
                strides_str,
                T::CUDA_TYPE,
                T::CUDA_TYPE,
                self.size(),
                self.ndim(),
            ),
            self.device(),
            &["contiguous"],
        )?;
        let kernel = res.device().get_func(&module_name, "contiguous").unwrap();
        let out_slice = res.cuda_slice();
        let inp_slice = self.cuda_slice();
        let reg_info = map.get("contiguous").expect("func_name not found");
        let cfg = compute_kernel_launch_config(res.device(), reg_info, res.size());
        unsafe { kernel.launch(cfg, (out_slice, inp_slice)) }?;
        Ok(res)
    }
}

impl<T, const DEVICE_ID: usize> TensorLike<T> for Tensor<T, Cuda, DEVICE_ID>
where
    T: CommonBounds + DeviceRepr,
{
    fn as_raw(&self) -> &[T] {
        unimplemented!()
    }

    fn as_raw_mut(&mut self) -> &mut [T] {
        unimplemented!()
    }

    fn contiguous(&self) -> anyhow::Result<Self> {
        Ok(self.inner.as_ref().contiguous()?.into())
    }
}

impl<T, const DEVICE_ID: usize> TensorInfo<T> for _Tensor<T, Cuda, DEVICE_ID>
where
    T: CommonBounds,
{
    fn ptr(&self) -> Pointer<T> {
        self.data.clone()
    }
    fn size(&self) -> usize {
        self.layout.size() as usize
    }
    fn shape(&self) -> &Shape {
        self.layout.shape()
    }
    fn strides(&self) -> &tensor_common::strides::Strides {
        self.layout.strides()
    }
    fn layout(&self) -> &Layout {
        &self.layout
    }
    fn parent(&self) -> Option<Pointer<T>> {
        self.parent.clone()
    }
    fn ndim(&self) -> usize {
        self.layout.ndim()
    }
    fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }
}

impl<T, const DEVICE_ID: usize> TensorInfo<T> for &_Tensor<T, Cuda, DEVICE_ID>
where
    T: CommonBounds,
{
    fn ptr(&self) -> Pointer<T> {
        self.data.clone()
    }

    fn size(&self) -> usize {
        self.layout.size() as usize
    }

    fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    fn strides(&self) -> &tensor_common::strides::Strides {
        self.layout.strides()
    }

    fn layout(&self) -> &Layout {
        &self.layout
    }

    fn parent(&self) -> Option<Pointer<T>> {
        self.parent.clone()
    }

    fn ndim(&self) -> usize {
        self.layout.ndim()
    }

    fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }
}

impl<T: CommonBounds + DeviceRepr, const DEVICE_ID: usize> TensorAlloc
    for _Tensor<T, Cuda, DEVICE_ID>
{
    type Meta = T;
    fn _empty<S: Into<Shape>>(shape: S) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        Self::empty(shape)
    }
}

// impl<T: CommonBounds> TensorIterator<'_, T> for _Tensor<T, Cuda> {}

impl<T: CommonBounds + DeviceRepr, const DEVICE_ID: usize> _Tensor<T, Cuda, DEVICE_ID> {
    // /// copy the data from the other tensor to this tensor
    // pub fn assign(&mut self, other: &_Tensor<T>) {
    //     self.par_iter_mut_simd()
    //         .zip(other.par_iter_simd())
    //         .for_each(|(a, b)| {
    //             *a = b;
    //         });
    // }

    /// cast the tensor to the new type
    pub fn astype<U>(&self) -> anyhow::Result<_Tensor<U, Cuda, DEVICE_ID>>
    where
        U: CommonBounds + DeviceRepr,
        T: IntoScalar<U>,
    {
        let mut ret: _Tensor<U, Cuda, DEVICE_ID> =
            _Tensor::<U, Cuda, DEVICE_ID>::empty(self.layout.shape().clone())?;
        uary_fn_with_out_simd(
            &ret,
            &get_module_name_1("astype", &ret),
            |out, idx| unimplemented!(),
            None::<_Tensor<U, Cuda, DEVICE_ID>>,
        )
    }

    // /// try to cast the tensor to the new type, if the type is the same, return the tensor itself, otherwise return the new tensor
    // pub fn try_astype<U>(&self) -> anyhow::Result<_Tensor<U, Cuda>>
    // where
    //     U: CommonBounds,
    //     T: IntoScalar<U>,
    // {
    //     if U::ID == T::ID {
    //         Ok(self.static_cast()?)
    //     } else {
    //         Ok(self.astype::<U>()?)
    //     }
    // }

    /// bitcast the tensor to the new type, the user must ensure the size of the new type is the same as the old type
    pub fn static_cast<Dst>(&self) -> anyhow::Result<_Tensor<Dst, Cuda, DEVICE_ID>>
    where
        Dst: CommonBounds,
    {
        if T::ID == Dst::ID {
            match self.parent.clone() {
                Some(parent) => {
                    #[cfg(feature = "bound_check")]
                    let new_parent = Pointer::new(parent.ptr as *mut Dst, parent.len);
                    #[cfg(not(feature = "bound_check"))]
                    let new_parent = Pointer::new(parent.ptr as *mut Dst);
                    Ok(_Tensor {
                        #[cfg(feature = "bound_check")]
                        data: Pointer::new(self.data.ptr as *mut Dst, self.ptr().len),
                        #[cfg(not(feature = "bound_check"))]
                        data: Pointer::new(self.data.ptr as *mut Dst),
                        parent: Some(new_parent),
                        mem_layout: self.mem_layout.clone(),
                        layout: self.layout.clone(),
                        _backend: self._backend.clone(),
                    })
                }
                None => Ok(_Tensor {
                    #[cfg(feature = "bound_check")]
                    data: Pointer::new(self.data.ptr as *mut Dst, self.ptr().len),
                    #[cfg(not(feature = "bound_check"))]
                    data: Pointer::new(self.data.ptr as *mut Dst),
                    parent: None,
                    mem_layout: self.mem_layout.clone(),
                    layout: self.layout.clone(),
                    _backend: self._backend.clone(),
                }),
            }
        } else {
            panic!("Cannot cast tensor to different type")
        }
    }

    // /// check if two tensors are close to each other
    // pub fn allclose<U: CommonBounds>(&self, other: &_Tensor<U, Cuda>) -> bool
    // where
    //     T: Convertor,
    //     U: Convertor,
    // {
    //     if self.shape() != other.shape() {
    //         return false;
    //     }
    //     let folder = self.par_iter().zip(other.par_iter()).fold(
    //         || true,
    //         |acc, (a, b)| {
    //             let a_val: f64 = a.to_f64();
    //             let b_val: f64 = b.to_f64();
    //             let abs_diff: f64 = (a_val - b_val).abs();
    //             let torlerance: f64 = 1.0e-8 + 1.0e-5 * b_val.abs();
    //             acc && abs_diff <= torlerance
    //         },
    //     );
    //     folder.reduce(|| true, |a, b| a && b)
    // }

    pub fn to_cpu(&self) -> anyhow::Result<Tensor<T>> {
        let mut data = _Tensor::<T>::empty(self.layout.shape().clone()).unwrap();
        let device = self.device();
        let ptr = unsafe { device.upgrade_device_ptr(self.data.ptr as u64, self.size()) };
        self.device()
            .dtoh_sync_copy_into(&ptr, data.as_raw_mut())
            .unwrap();
        ptr.leak();
        Ok(data.into())
    }
    pub(crate) fn device(&self) -> Arc<CudaDevice> {
        self._backend._backend.device.clone()
    }
    pub(crate) fn cuda_slice(&self) -> super::cuda_slice::CudaSlice {
        super::cuda_slice::CudaSlice {
            inner: self.data.ptr as u64,
        }
    }
    pub(crate) fn cuda_shape(&self) -> anyhow::Result<cudarc::driver::CudaSlice<i64>> {
        Ok(self.device().htod_sync_copy(self.shape())?)
    }
    pub(crate) fn cuda_strides(&self) -> anyhow::Result<cudarc::driver::CudaSlice<i64>> {
        Ok(self.device().htod_sync_copy(self.strides())?)
    }
    pub(crate) fn device_cap(&self) -> usize {
        self._backend._backend.cap
    }
}

impl<T: CommonBounds + DeviceRepr, const DEVICE_ID: usize> Tensor<T, Cuda, DEVICE_ID> {
    /// copy the data from the cuda tensor to the cpu tensor
    pub fn to_cpu(&self) -> anyhow::Result<Tensor<T>> {
        Ok(self.inner.as_ref().to_cpu()?.into())
    }
    /// get the device of the tensor
    pub fn device(&self) -> Arc<CudaDevice> {
        self.inner.as_ref().device()
    }

    // /// copy the data from the other tensor to this tensor
    // pub fn assign(&mut self, other: &Tensor<T, Cuda>) {
    //     let mut mut_self = self.inner.as_ref().clone();
    //     mut_self.assign(&other.inner.as_ref());
    // }

    // /// cast the tensor to the new type
    // pub fn astype<U>(&self) -> anyhow::Result<Tensor<U, Cuda>>
    // where
    //     U: CommonBounds,
    //     T: IntoScalar<U>,
    // {
    //     Ok(self.inner.astype()?.into())
    // }

    // /// try to cast the tensor to the new type, if the type is the same, return the tensor itself, otherwise return the new tensor
    // pub fn try_astype<U>(&self) -> anyhow::Result<Tensor<U, Cuda>>
    // where
    //     U: CommonBounds,
    //     T: IntoScalar<U>,
    // {
    //     Ok(self.inner.try_astype()?.into())
    // }

    // /// bitcast the tensor to the new type, the user must ensure the size of the new type is the same as the old type
    // pub fn static_cast<Dst>(&self) -> anyhow::Result<Tensor<Dst, Cuda>>
    // where
    //     Dst: CommonBounds,
    // {
    //     Ok(self.inner.static_cast()?.into())
    // }

    // /// check if two tensors are close to each other
    // pub fn allclose<U: CommonBounds>(&self, other: &Tensor<U, Cuda>) -> bool
    // where
    //     T: Convertor,
    //     U: Convertor,
    // {
    //     self.inner.allclose(&other.inner)
    // }
}

impl<T, const DEVICE_ID: usize> std::fmt::Display for _Tensor<T, Cuda, DEVICE_ID>
where
    T: CommonBounds + Convertor + DeviceRepr,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut data = _Tensor::<T>::empty(self.layout.shape().clone()).unwrap();
        let device = self.device();
        let ptr = unsafe { device.upgrade_device_ptr(self.ptr().ptr as u64, self.size()) };
        self.device()
            .dtoh_sync_copy_into(&ptr, data.as_raw_mut())
            .unwrap();
        ptr.leak();
        data.layout.set_strides(self.strides().clone());
        data.layout.set_shape(self.shape().clone());
        write!(f, "{}", data)
    }
}

impl<T, const DEVICE_ID: usize> std::fmt::Display for Tensor<T, Cuda, DEVICE_ID>
where
    T: CommonBounds + Convertor + DeviceRepr,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner.as_ref())
    }
}

impl<T, const DEVICE_ID: usize> Into<Tensor<T, Cuda, DEVICE_ID>> for _Tensor<T, Cuda, DEVICE_ID> {
    fn into(self) -> Tensor<T, Cuda, DEVICE_ID> {
        Tensor { inner: self.into() }
    }
}

impl<T, const DEVICE_ID: usize> Into<Tensor<T, Cuda, DEVICE_ID>> for &Tensor<T, Cuda, DEVICE_ID> {
    fn into(self) -> Tensor<T, Cuda, DEVICE_ID> {
        Tensor {
            inner: self.inner.clone(),
        }
    }
}

impl<T, const DEVICE_ID: usize> From<T> for _Tensor<T, Cuda, DEVICE_ID>
where
    T: DeviceRepr + CommonBounds,
{
    fn from(value: T) -> Self {
        if let Ok(mut cache) = CUDA_CACHE.lock() {
            let (ptr, device) = cache.host_to_device(&[value], DEVICE_ID).unwrap();
            let layout = Layout::new(vec![1], vec![1]);
            let mem_layout = std::alloc::Layout::from_size_align(size_of::<T>(), ALIGN).unwrap();
            _Tensor {
                #[cfg(feature = "bound_check")]
                data: Pointer::new(ptr as *mut T, 1),
                #[cfg(not(feature = "bound_check"))]
                data: Pointer::new(ptr as *mut T),
                parent: None,
                layout,
                mem_layout: Arc::new(mem_layout),
                _backend: Backend::<Cuda>::new(ptr as u64, device),
            }
        } else {
            panic!("Failed to lock CUDA cache");
        }
    }
}

impl<'a, T, const DEVICE_ID: usize> From<&'a T> for _Tensor<T, Cuda, DEVICE_ID>
where
    T: DeviceRepr + CommonBounds,
{
    fn from(value: &'a T) -> Self {
        _Tensor::from(*value)
    }
}

// impl<'a, T> Into<_Tensor<T, Cuda>> for &'a [T] {
//     fn into(self) -> _Tensor<T, Cuda> {
//         let shape = vec![self.len() as i64];
//         let strides = vec![1];
//         let layout = Layout::new(shape, strides);
//         let mem_layout =
//             std::alloc::Layout::from_size_align(self.len() * size_of::<T>(), ALIGN).unwrap();
//         let ptr = CACHE.allocate(mem_layout.clone()).unwrap();
//         unsafe {
//             std::ptr::copy_nonoverlapping(self.as_ptr(), ptr as *mut T, self.len());
//         }
//         _Tensor {
//             #[cfg(feature = "bound_check")]
//             data: Pointer::new(ptr as *mut T, self.len() as i64),
//             #[cfg(not(feature = "bound_check"))]
//             data: Pointer::new(ptr as *mut T),
//             parent: None,
//             layout,
//             mem_layout: Arc::new(mem_layout),
//             _backend: Backend::new(ptr as u64),
//         }
//     }
// }
