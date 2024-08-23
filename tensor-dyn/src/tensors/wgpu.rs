#![allow(unused)]
use std::{ fmt::{ Display, Debug }, ops::{ Div, Mul, Sub }, sync::Arc };
use bytemuck::Pod;
use wgpu::util::DeviceExt;
use rand_distr::{
    uniform::SampleUniform,
    Distribution,
    Exp1,
    Open01,
    OpenClosed01,
    Standard,
    StandardNormal,
};
use tensor_allocator::{ CACHE, WGPU_CACHE };
use tensor_common::{
    axis::{ process_axes, Axis },
    err_handler::ErrHandler,
    layout::Layout,
    pointer::Pointer,
    shape::Shape,
    shape_utils::{ yield_one_after, yield_one_before },
    slice::Slice,
};
use tensor_display::display;
use tensor_macros::match_selection;
use tensor_common::slice;
use tensor_iterator::{ par_strided::ParStrided, par_strided_mut::ParStridedMut };
use tensor_traits::{
    random::Random,
    shape_manipulate::ShapeManipulate,
    tensor::{ CommonBounds, TensorAlloc, TensorCreator, TensorInfo, TensorLike },
    BaseTensor,
};
use tensor_common::shape_utils::try_pad_shape;
use anyhow::Result;
use tensor_types::{
    convertion::{ Convertor, FromScalar },
    dtype::Dtype,
    into_scalar::IntoScalar,
    type_promote::{ FloatOut, NormalOut },
};
use rayon::iter::{
    IndexedParallelIterator,
    IntoParallelIterator,
    IntoParallelRefIterator,
    IntoParallelRefMutIterator,
    ParallelIterator,
};

use crate::{
    backend::{ Backend, BackendDevice, BackendTy, Cpu, Wgpu },
    ops::{ cpu::concat::concat, wgpu::buffer_helper::WgpuDevice },
    slice::SliceOps,
    tensor::Tensor,
    tensor_base::_Tensor, ALIGN,
};

impl<T> TensorInfo<T> for _Tensor<T, Wgpu> where T: CommonBounds {
    fn ptr(&self) -> Pointer<T> {
        todo!()
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
        self.parent
    }

    fn ndim(&self) -> usize {
        self.layout.ndim()
    }

    fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }
}

impl<T> TensorInfo<T> for &_Tensor<T, Wgpu> where T: CommonBounds {
    fn ptr(&self) -> Pointer<T> {
        todo!()
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
        self.parent
    }

    fn ndim(&self) -> usize {
        self.layout.ndim()
    }

    fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }
}

impl<T: CommonBounds> BaseTensor for _Tensor<T, Wgpu> {
    type Output = _Tensor<T, Wgpu>;
    fn base(&self) -> &Self::Output {
        &self
    }
}

impl<T: CommonBounds> BaseTensor for &_Tensor<T, Wgpu> {
    type Output = _Tensor<T, Wgpu>;
    fn base(&self) -> &Self::Output {
        &self
    }
}

impl<T: CommonBounds> _Tensor<T, Wgpu> {
    pub fn to_cpu(&self) -> _Tensor<T, Cpu> where T: Pod + Debug {
        let res_size = self.size() * std::mem::size_of::<T>();
        let read_buffer = self.device().create_buffer(
            &(wgpu::BufferDescriptor {
                label: Some("Result Buffer"),
                size: res_size as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        );
        let mut encoder = self
            .device()
            .create_command_encoder(&(wgpu::CommandEncoderDescriptor { label: None }));
        encoder.copy_buffer_to_buffer(self.buffer(), 0, &read_buffer, 0, res_size as u64);
        self.device().queue.submit(Some(encoder.finish()));
        let buffer_slice = read_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device().poll(wgpu::Maintain::wait()).panic_on_timeout();
        let tensor = _Tensor::<T, Cpu>::empty(self.shape()).unwrap();
        pollster::block_on(async {
            if let Ok(Ok(())) = receiver.recv_async().await {
                let data = buffer_slice.get_mapped_range();
                let result: &[T] = bytemuck::cast_slice(&data);
                println!("{:?}", result);
                tensor
                    .as_raw_mut()
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(i, x)| {
                        *x = result[i];
                    });
            }
        });
        tensor
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self._backend._backend.buffer.buffer
    }

    pub fn device(&self) -> &WgpuDevice {
        &self._backend._backend.device
    }

    /// Converts a tensor to a raw slice representing direct memory access.
    ///
    /// This function provides direct, read-only access to the tensor's underlying memory. It is useful
    /// for interfacing with low-level or external functions that require direct memory access.
    ///
    /// # Returns
    /// `&[T]`: A raw slice providing direct, read-only access to the tensor's memory.
    ///
    /// # Caution
    /// - Direct memory access can lead to undefined behavior if not handled properly.
    /// - This function bypasses Rust's safety checks, so caution must be exercised to avoid data corruption
    ///   or undefined behavior.
    /// - The caller is responsible for ensuring that the memory accessed is valid and not being mutated
    ///   elsewhere concurrently.
    ///
    /// # Examples
    /// ```
    /// let tensor = YourType::new(...);
    /// let direct_memory_access = tensor.as_raw();
    /// // Use direct_memory_access for operations requiring direct memory access
    /// ```
    pub fn as_raw(&self) -> &[T] {
        todo!()
    }

    /// Converts a tensor to a raw mutable slice representing direct memory access.
    ///
    /// This function provides direct, mutable access to the tensor's underlying memory. It is intended for
    /// advanced use cases where direct memory manipulation is required.
    ///
    /// # Returns
    /// `&mut [T]`: A raw mutable slice providing direct access to the tensor's memory.
    ///
    /// # Caution
    /// - Modifying data through this interface can lead to undefined behavior and should be done with utmost care.
    /// - This method bypasses Rust's safety and concurrency checks.
    /// - The caller must ensure that no other references to the tensor data are being mutated concurrently.
    ///
    /// # Examples
    /// ```
    /// let mut tensor = YourType::new(...);
    /// let direct_memory_access_mut = tensor.as_raw_mut();
    /// // Perform operations requiring direct and mutable memory access
    /// ```
    pub fn as_raw_mut(&self) -> &mut [T] {
        todo!()
    }

    pub fn iter(&self) -> ParStrided<T> {
        todo!()
    }

    pub fn iter_mut(&self) -> ParStridedMut<T> {
        todo!()
    }

    /// Converts the tensor to a new type.
    ///
    /// This method attempts to convert the elements of the tensor to a specified type `U`.
    /// It returns a new tensor with the converted elements.
    ///
    /// # Returns
    /// A `Result` containing the new tensor of type `U` or an error if the conversion fails.
    ///
    /// # Examples
    /// ```
    /// use tensor_core::_Tensor;
    /// let tensor = _Tensor::<f32>::new([1.0, 2.0, 3.0]);
    /// let converted_tensor = tensor.astype::<i32>().unwrap();
    /// assert!(tensor.allclose(&converted_tensor))
    /// ```
    pub fn astype<U>(&self) -> Result<_Tensor<U, Wgpu>>
        where U: CommonBounds + Pod, T: IntoScalar<U>
    {
        // Create an empty tensor of the new type with the same shape.
        let ret: _Tensor<U, Wgpu> = _Tensor::<U, Wgpu>::empty(
            self.layout.shape().clone(),
            &self._backend._backend.device
        )?;

        // Parallel iteration to convert and copy each element to the new tensor.
        todo!()
    }

    /// Try to cast the tensor to a new type, with an optimization for same-type casting.
    ///
    /// This method checks if the target type `U` is the same as the current type `T`.
    /// If they are the same, it performs a static cast, which is more efficient.
    /// Otherwise, it falls back to the `astype` method.
    /// Note that type checking is done at compile time.
    ///
    /// # Returns
    /// A `Result` containing the new tensor of type `U` or an error if the conversion fails.
    ///
    /// # Examples
    /// ```
    /// use tensor_core::_Tensor;
    /// let tensor = _Tensor::<f32>::new([1.0, 2.0, 3.0]);
    /// let converted_tensor = tensor.try_astype::<i32>().unwrap();
    /// assert!(tensor.allclose(&converted_tensor))
    /// ```
    pub fn try_astype<U>(&self) -> Result<_Tensor<U, Wgpu>>
        where U: CommonBounds + Pod, T: IntoScalar<U>
    {
        if U::ID == T::ID { Ok(self.static_cast()?) } else { Ok(self.astype::<U>()?) }
    }

    /// Performs a static cast of the tensor to a new type without actual data conversion.
    ///
    /// This method is used when the target type `U` is the same as the current type `T`.
    /// It reinterprets the tensor data as the new type without changing the underlying bytes.
    ///
    /// # Returns
    /// A `Result` containing the new tensor of type `U`.
    ///
    /// # Examples
    /// ```
    /// use tensor_core::_Tensor;
    /// let tensor = _Tensor::<f32>::new([1.0, 2.0, 3.0]);
    /// let static_cast_tensor = tensor.static_cast::<f32>().unwrap();
    /// assert!(tensor.allclose(&static_cast_tensor))
    /// ```
    pub fn static_cast<U>(&self) -> Result<_Tensor<U, Wgpu>> where U: CommonBounds {
        assert_eq!(U::ID, T::ID);
        match self.parent {
            Some(parent) => {
                let new_parent = Pointer::new(parent.ptr as *mut U);
                Ok(_Tensor {
                    data: Pointer::new(self.data.ptr as *mut U),
                    parent: Some(new_parent),
                    mem_layout: self.mem_layout.clone(),
                    layout: self.layout.clone(),
                    _backend: self._backend.clone(),
                })
            }
            None => {
                let new_parent = Pointer::new(self.data.ptr as *mut U);
                Ok(_Tensor {
                    data: Pointer::new(self.data.ptr as *mut U),
                    parent: Some(new_parent),
                    mem_layout: self.mem_layout.clone(),
                    layout: self.layout.clone(),
                    _backend: self._backend.clone(),
                })
            }
        }
    }

    /// Checks if all elements of the tensor are close to the elements of another tensor.
    ///
    /// This method computes the absolute difference between each pair of corresponding elements
    /// and checks if they are within a specified tolerance. The default tolerance values
    /// are `1.0e-8` and `1.0e-5`.
    ///
    /// # Arguments
    /// - `other`: The other tensor to compare with.
    ///
    /// # Returns
    /// `true` if all elements are close; otherwise, `false`.
    ///
    /// # Examples
    /// ```
    /// use tensor_core::_Tensor;
    /// let tensor1 = _Tensor::<f64>::new([1.0, 2.0, 3.0]);
    /// let tensor2 = _Tensor::<f64>::new([1.0, 2.0, 3.0]);
    /// assert!(tensor1.allclose(&tensor2));
    /// ```
    pub fn allclose<U: CommonBounds>(&self, other: &_Tensor<U>) -> bool
        where T: Convertor, U: Convertor
    {
        if self.shape() != other.shape() {
            return false;
        }
        let folder = self
            .iter()
            .zip(other.par_iter())
            .fold(
                || true,
                |acc, (a, b)| {
                    let a_val: f64 = a.to_f64();
                    let b_val: f64 = b.to_f64();
                    let abs_diff: f64 = (a_val - b_val).abs();
                    let torlerance: f64 = 1.0e-8 + 1.0e-5 * b_val.abs();
                    acc && abs_diff <= torlerance
                }
            );
        folder.reduce(
            || true,
            |a, b| a && b
        )
    }

    /// Create a contiguous copy of the tensor.
    ///
    /// This method returns a new tensor that has the same data as the original tensor,
    /// but with elements laid out in contiguous memory.
    ///
    /// # Returns
    /// A `Result` containing the new contiguous tensor.
    ///
    /// # Examples
    /// ```
    /// use tensor_core::_Tensor;
    /// use tensor_trait::{ShapeManipulate, TensorInfo};
    /// let tensor = _Tensor::<f32>::new([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
    /// let transposed_tensor = tensor.transpose([1, 0]).unwrap();
    /// let contiguous_tensor = tensor.contiguous().unwrap();
    /// assert!(contiguous_tensor.is_contiguous())
    /// ```
    pub fn contiguous(&self) -> Result<Self> {
        todo!()
    }

    /// Stacks a sequence of tensors along a specified axis.
    ///
    /// Given a list of tensors, this function concatenates them along the specified axis.
    /// All tensors must have the same shape, except in the dimension corresponding to `axis`.
    ///
    /// # Arguments
    /// - `tensors`: A vector of tensor references to be stacked.
    /// - `axis`: The axis along which the tensors will be stacked.
    /// - `keepdims`: A boolean indicating whether to keep the dimension of the axis or not.
    ///
    /// # Returns
    /// A `Result` containing the stacked tensor or an error if the operation fails.
    ///
    /// # Examples
    /// ```
    /// use tensor_core::_Tensor;
    /// let tensor1 = _Tensor::<f32>::new([1.0, 2.0, 3.0]);
    /// let tensor2 = _Tensor::<f32>::new([4.0, 5.0, 6.0]);
    /// let stacked_tensor = _Tensor::stack(vec![&tensor1, &tensor2], 0, true).unwrap();
    /// assert!(stacked_tensor.allclose(&_Tensor::<f64>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])));
    /// ```
    pub fn stack(tensors: Vec<&_Tensor<T, Wgpu>>, axis: usize, keepdims: bool) -> Result<Self>
        where T: 'static
    {
        todo!()
    }

    /// Vertically stacks a sequence of tensors.
    ///
    /// This is a convenience method for stacking tensors along the first axis (axis=0).
    /// All tensors must have the same number of dimensions and the same shape,
    /// except for the first axis.
    ///
    /// # Arguments
    /// - `tensors`: A vector of tensor references to be vertically stacked.
    ///
    /// # Returns
    /// A `Result` containing the vertically stacked tensor or an error if the operation fails.
    ///
    /// # Examples
    /// ```
    /// use tensor_core::_Tensor;
    /// let tensor1 = _Tensor::<f32>::new([1.0, 2.0, 3.0]);
    /// let tensor2 = _Tensor::<f32>::new([4.0, 5.0, 6.0]);
    /// let vstacked_tensor = _Tensor::vstack(vec![&tensor1, &tensor2]).unwrap();
    /// assert!(vstacked_tensor.allclose(&_Tensor::<f64>::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])));
    /// ```
    pub fn vstack(tensors: Vec<&_Tensor<T, Wgpu>>) -> Result<_Tensor<T>> {
        todo!()
    }
    /// Horizontally stacks a sequence of tensors.
    ///
    /// This function concatenates tensors along the second axis (axis=1).
    /// It automatically reshapes tensors with fewer dimensions to have an additional axis.
    /// For 1-dimensional tensors, they are reshaped to 2D before stacking.
    /// Scalars are reshaped to 1x1 tensors.
    ///
    /// # Arguments
    /// - `tensors`: A vector of references to the tensors to be horizontally stacked.
    ///
    /// # Returns
    /// A `Result` containing the horizontally stacked tensor or an error if the operation fails.
    ///
    /// # Examples
    /// ```
    /// use tensor_core::_Tensor;
    /// let tensor1 = _Tensor::<f32>::new([1.0, 2.0, 3.0]);
    /// let tensor2 = _Tensor::<f32>::new([4.0, 5.0, 6.0]);
    /// let hstacked_tensor = _Tensor::hstack(vec![&tensor1, &tensor2]).unwrap();
    /// assert!(hstacked_tensor.allclose(&_Tensor::<f64>::new([1.0, 2.0, 3.0,4.0, 5.0, 6.0])));
    /// ```
    pub fn hstack(mut tensors: Vec<&_Tensor<T, Wgpu>>) -> Result<_Tensor<T, Wgpu>> {
        todo!()
    }

    /// Depth-stacks a sequence of tensors.
    ///
    /// This function concatenates tensors along the third axis (axis=2).
    /// It automatically reshapes tensors with fewer dimensions to match the required number of dimensions.
    /// For 1-dimensional tensors, they are reshaped to 1xNx1 before stacking.
    /// For 2-dimensional tensors, they are reshaped to NxMx1.
    /// Scalars are reshaped to 1x1x1 tensors.
    ///
    /// # Arguments
    /// - `tensors`: A vector of references to the tensors to be depth-stacked.
    ///
    /// # Returns
    /// A `Result` containing the depth-stacked tensor or an error if the operation fails.
    ///
    /// # Examples
    /// ```
    /// use tensor_core::_Tensor;
    /// let tensor1 = _Tensor::<f32>::new([1.0, 2.0, 3.0]);
    /// let tensor2 = _Tensor::<f32>::new([4.0, 5.0, 6.0]);
    /// let dstacked_tensor = _Tensor::dstack(vec![&tensor1, &tensor2]).unwrap();
    /// assert!(dstacked_tensor.allclose(&_Tensor::<f64>::new([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]])));
    /// ```
    pub fn dstack(mut tensors: Vec<&_Tensor<T, Wgpu>>) -> Result<_Tensor<T, Wgpu>> {
        todo!()
    }
}

impl<T: CommonBounds + Pod> _Tensor<T, Wgpu> {
    pub fn empty<S: Into<Shape>>(shape: S, device: &WgpuDevice) -> Result<Self> {
        let _shape = shape.into();
        let res_shape = Shape::from(_shape);
        let mut size = 1;
        let mut strides = vec![0; res_shape.len()];
        for i in (0..res_shape.len()).rev() {
            let tmp = res_shape[i];
            strides[i] = size as i64;
            size *= tmp as usize;
        }
        let layout = std::alloc::Layout
            ::from_size_align(size * std::mem::size_of::<T>(), ALIGN)
            .unwrap();

        let buffer = unsafe { WGPU_CACHE.allocate(&layout, &device.device, false) };

        let backend = Backend {
            _backend: Wgpu {
                buffer,
                device: device.clone(),
            },
        };
        Ok(_Tensor {
            data: Pointer::new(std::ptr::null_mut()),
            parent: None,
            mem_layout: layout.into(),
            layout: Layout::new(res_shape, strides),
            _backend: backend,
        })
    }

    fn zeros<S: Into<Shape>>(shape: S, device: &WgpuDevice) -> Result<Self> {
        let zeros = _Tensor::<T, Cpu>::zeros(shape)?;
        let layout = &zeros.mem_layout;
        let buffer = unsafe { WGPU_CACHE.allocate(layout, &device.device, false) };
        let mut encoder = device.device.device.create_command_encoder(
            &(wgpu::CommandEncoderDescriptor {
                label: Some("Command Encoder"),
            })
        );
        let staging_buffer = device.device.device.create_buffer_init(
            &(wgpu::util::BufferInitDescriptor {
                label: Some("Staging Buffer"),
                contents: bytemuck::cast_slice(&vec![0u8; zeros.size() * std::mem::size_of::<T>()]),
                usage: wgpu::BufferUsages::COPY_SRC,
            })
        );
        encoder.copy_buffer_to_buffer(
            &staging_buffer,
            0,
            &buffer.buffer,
            0,
            (zeros.size() * std::mem::size_of::<T>()) as wgpu::BufferAddress
        );
        device.queue.submit(Some(encoder.finish()));
        let backend = Backend {
            _backend: Wgpu {
                buffer,
                device: device.clone(),
            },
        };
        Ok(_Tensor {
            data: Pointer::new(std::ptr::null_mut()),
            parent: None,
            mem_layout: zeros.mem_layout.clone(),
            layout: zeros.layout.clone(),
            _backend: backend,
        })
    }

    fn ones<S: Into<Shape>>(shape: S, device: &WgpuDevice) -> Result<Self> where u8: IntoScalar<T> {
        let ones = _Tensor::<T, Cpu>::ones(shape)?;
        let layout = &ones.mem_layout;
        let buffer = unsafe { WGPU_CACHE.allocate(layout, &device.device, false) };
        let mut encoder = device.device.device.create_command_encoder(
            &(wgpu::CommandEncoderDescriptor {
                label: Some("Command Encoder"),
            })
        );
        let staging_buffer = device.device.device.create_buffer_init(
            &(wgpu::util::BufferInitDescriptor {
                label: Some("Staging Buffer"),
                contents: bytemuck::cast_slice(&vec![1u8; ones.size() * std::mem::size_of::<T>()]),
                usage: wgpu::BufferUsages::COPY_SRC,
            })
        );
        encoder.copy_buffer_to_buffer(
            &staging_buffer,
            0,
            &buffer.buffer,
            0,
            (ones.size() * std::mem::size_of::<T>()) as wgpu::BufferAddress
        );
        device.queue.submit(Some(encoder.finish()));
        let backend = Backend {
            _backend: Wgpu {
                buffer,
                device: device.clone(),
            },
        };
        Ok(_Tensor {
            data: Pointer::new(std::ptr::null_mut()),
            parent: None,
            mem_layout: layout.clone(),
            layout: ones.layout.clone(),
            _backend: backend,
        })
    }

    fn empty_like(&self) -> Result<Self> {
        Self::empty(self.shape(), &self._backend._backend.device)
    }

    fn zeros_like(&self) -> Result<Self> {
        Self::zeros(self.shape(), &self._backend._backend.device)
    }

    fn ones_like(&self) -> Result<Self> where u8: IntoScalar<T> {
        Self::ones(self.shape(), &self._backend._backend.device)
    }

    fn full<S: Into<Shape>>(val: T, shape: S) -> Result<Self> {
        todo!()
    }

    fn full_like(&self, val: T) -> Result<Self> {
        _Tensor::full(val, self.shape())
    }

    pub async fn arange<'a, U>(start: U, end: U, device: &WgpuDevice) -> Result<Self>
        where T: Convertor + FromScalar<U> + NormalOut<T, Output = T>, usize: IntoScalar<T>
    {
        let arange = _Tensor::<T, Cpu>::arange(start, end)?;
        let layout = &arange.mem_layout;
        let buffer = unsafe { WGPU_CACHE.allocate(layout, &device.device, false) };

        device.queue.write_buffer(&buffer.buffer, 0, bytemuck::cast_slice(arange.as_raw()));

        let backend = Backend {
            _backend: Wgpu {
                buffer,
                device: device.clone(),
            },
        };
        Ok(_Tensor {
            data: Pointer::new(std::ptr::null_mut()),
            parent: None,
            mem_layout: arange.mem_layout.clone(),
            layout: arange.layout.clone(),
            _backend: backend,
        })
    }

    fn arange_step(start: T, end: T, step: T, device: &WgpuDevice) -> Result<Self>
        where T: Convertor + FromScalar<usize> + NormalOut<T, Output = T>
    {
        let arange = _Tensor::<T, Cpu>::arange_step(start, end, step)?;
        let layout = &arange.mem_layout;
        let buffer = unsafe { WGPU_CACHE.allocate(layout, &device.device, false) };
        let mut encoder = device.device.device.create_command_encoder(
            &(wgpu::CommandEncoderDescriptor {
                label: Some("Command Encoder"),
            })
        );
        let staging_buffer = device.device.device.create_buffer_init(
            &(wgpu::util::BufferInitDescriptor {
                label: Some("Staging Buffer"),
                contents: bytemuck::cast_slice(arange.as_raw()),
                usage: wgpu::BufferUsages::COPY_SRC,
            })
        );
        encoder.copy_buffer_to_buffer(
            &staging_buffer,
            0,
            &buffer.buffer,
            0,
            (arange.size() * std::mem::size_of::<T>()) as wgpu::BufferAddress
        );
        device.queue.submit(Some(encoder.finish()));
        let backend = Backend {
            _backend: Wgpu {
                buffer,
                device: device.clone(),
            },
        };
        Ok(_Tensor {
            data: Pointer::new(std::ptr::null_mut()),
            parent: None,
            mem_layout: arange.mem_layout.clone(),
            layout: arange.layout.clone(),
            _backend: backend,
        })
    }

    fn eye(n: usize, m: usize, k: usize, device: &WgpuDevice) -> Result<Self>
        where u8: IntoScalar<T>
    {
        let eye = _Tensor::<T, Cpu>::eye(n, m, k)?;
        let layout = &eye.mem_layout;
        let buffer = unsafe { WGPU_CACHE.allocate(layout, &device.device, false) };
        let mut encoder = device.device.device.create_command_encoder(
            &(wgpu::CommandEncoderDescriptor {
                label: Some("Command Encoder"),
            })
        );
        let staging_buffer = device.device.device.create_buffer_init(
            &(wgpu::util::BufferInitDescriptor {
                label: Some("Staging Buffer"),
                contents: bytemuck::cast_slice(eye.as_raw()),
                usage: wgpu::BufferUsages::COPY_SRC,
            })
        );
        encoder.copy_buffer_to_buffer(
            &staging_buffer,
            0,
            &buffer.buffer,
            0,
            (eye.size() * std::mem::size_of::<T>()) as wgpu::BufferAddress
        );
        device.queue.submit(Some(encoder.finish()));
        let backend = Backend {
            _backend: Wgpu {
                buffer,
                device: device.clone(),
            },
        };
        Ok(_Tensor {
            data: Pointer::new(std::ptr::null_mut()),
            parent: None,
            mem_layout: eye.mem_layout.clone(),
            layout: eye.layout.clone(),
            _backend: backend,
        })
    }

    fn linspace(start: T, end: T, num: usize, include_end: bool) -> Result<Self>
        where
            T: Convertor + num::Float + NormalOut<T, Output = T>,
            usize: IntoScalar<T>,
            f64: IntoScalar<T>
    {
        todo!()
    }

    fn logspace(start: T, end: T, num: usize, include_end: bool, base: T) -> Result<Self>
        where
            T: Convertor +
                num::Float +
                FromScalar<usize> +
                FromScalar<f64> +
                NormalOut<T, Output = T>
    {
        todo!()
    }

    fn geomspace(start: T, end: T, n: usize, include_end: bool) -> Result<Self>
        where
            T: PartialOrd +
                FloatOut<T> +
                NormalOut<T, Output = T> +
                FromScalar<<T as FloatOut>::Output> +
                std::ops::Neg<Output = T>,
            <T as FloatOut>::Output: Sub<Output = <T as FloatOut>::Output> +
                FromScalar<usize> +
                FromScalar<f64> +
                Div<Output = <T as FloatOut>::Output> +
                NormalOut<Output = <T as FloatOut>::Output> +
                CommonBounds
    {
        todo!()
    }

    fn tri(n: usize, m: usize, k: i64, low_triangle: bool, device: &WgpuDevice) -> Result<Self>
        where u8: IntoScalar<T>
    {
        let tri = _Tensor::<T, Cpu>::tri(n, m, k, low_triangle)?;
        let layout = &tri.mem_layout;
        let buffer = unsafe { WGPU_CACHE.allocate(layout, &device.device, false) };
        let mut encoder = device.device.device.create_command_encoder(
            &(wgpu::CommandEncoderDescriptor {
                label: Some("Command Encoder"),
            })
        );
        let staging_buffer = device.device.device.create_buffer_init(
            &(wgpu::util::BufferInitDescriptor {
                label: Some("Staging Buffer"),
                contents: bytemuck::cast_slice(tri.as_raw()),
                usage: wgpu::BufferUsages::COPY_SRC,
            })
        );
        encoder.copy_buffer_to_buffer(
            &staging_buffer,
            0,
            &buffer.buffer,
            0,
            (tri.size() * std::mem::size_of::<T>()) as wgpu::BufferAddress
        );
        device.queue.submit(Some(encoder.finish()));
        let backend = Backend {
            _backend: Wgpu {
                buffer,
                device: device.clone(),
            },
        };
        Ok(_Tensor {
            data: Pointer::new(std::ptr::null_mut()),
            parent: None,
            mem_layout: tri.mem_layout.clone(),
            layout: tri.layout.clone(),
            _backend: backend,
        })
    }

    fn tril(&self, k: i64) -> Result<Self> where T: NormalOut<bool, Output = T> + IntoScalar<T> {
        todo!()
    }

    fn triu(&self, k: i64) -> Result<Self> where T: NormalOut<bool, Output = T> + IntoScalar<T> {
        todo!()
    }

    fn identity(n: usize, device: &WgpuDevice) -> Result<Self> where u8: IntoScalar<T> {
        _Tensor::eye(n, n, 0, device)
    }
}

impl<T: CommonBounds> ShapeManipulate for _Tensor<T, Wgpu> {
    type Meta = T;
    fn squeeze<A: Into<Axis>>(&self, axes: A) -> Result<_Tensor<T, Wgpu>> {
        let axes: Vec<usize> = process_axes(axes, self.ndim())?;
        for i in 0..axes.len() {
            if self.shape()[axes[i]] != 1 {
                return Err(
                    anyhow::anyhow!(
                        "cannot select an axis to squeeze out which has size not equal to one, try to squeeze axis {} with size {} in shape {:?}",
                        axes[i],
                        self.shape()[axes[i]],
                        self.shape()
                    )
                );
            }
        }
        let new_shape: Vec<i64> = self
            .shape()
            .iter()
            .enumerate()
            .filter(|&(i, _)| !axes.contains(&i))
            .map(|(_, &x)| x)
            .collect();
        self.reshape(new_shape)
    }

    fn unsqueeze<A: Into<Axis>>(&self, axes: A) -> Result<_Tensor<T, Wgpu>> {
        let mut res_shape: Vec<i64> = self.shape().to_vec();
        let axes: Vec<usize> = process_axes(axes, self.ndim())?;
        axes.iter().for_each(|&x| {
            res_shape = yield_one_before(&res_shape, x);
        });
        self.reshape(res_shape)
    }

    fn reshape<S: Into<Shape>>(&self, shape: S) -> Result<_Tensor<T, Wgpu>> {
        let shape = shape.into();
        ErrHandler::check_size_match(shape.size(), self.shape().size())?;
        if let Ok(new_layout) = self.layout.inplace_reshape(&shape) {
            Ok(_Tensor {
                data: self.data.clone(),
                parent: self.parent.clone(),
                mem_layout: self.mem_layout.clone(),
                layout: new_layout,
                _backend: self._backend.clone(),
            })
        } else {
            todo!()
        }
    }

    fn transpose(&self, axis1: i64, axis2: i64) -> Result<_Tensor<T, Wgpu>> {
        if self.ndim() < 2 {
            Err(
                anyhow::Error::msg(
                    "_Tensor with less than 2 dimensions for `transpose` method is not allowed"
                )
            )
        } else {
            self.permute(vec![axis1, axis2])
        }
    }

    fn permute<A: Into<Axis>>(&self, axes: A) -> Result<_Tensor<T, Wgpu>> {
        let permuted_layout = self.layout.permute(axes)?;
        todo!()
    }

    fn expand<S: Into<Shape>>(&self, shape: S) -> Result<_Tensor<T, Wgpu>> {
        let res_shape = Shape::from(shape.into());
        let res_strides = self.layout.expand_strides(&res_shape);
        todo!()
    }

    fn t(&self) -> Result<Self> {
        if self.ndim() < 2 {
            return Err(
                anyhow::Error::msg(
                    "_Tensor with less than 2 dimensions for `t` method is not allowed"
                )
            );
        } else if self.ndim() > 2 {
            let mut axes = (0..self.ndim() as i64).collect::<Vec<i64>>();
            axes.swap(self.ndim() - 1, self.ndim() - 2);
            return self.permute(axes);
        }
        self.transpose(1, 0)
    }

    fn mt(&self) -> Result<Self> {
        self.permute((0..self.ndim() as i64).rev().collect::<Vec<i64>>())
    }

    fn flip<A: Into<Axis>>(&self, axes: A) -> Result<Self> {
        let axes = process_axes(axes, self.ndim())?;
        let mut new_strides = self.strides().to_vec();
        let mut ptr = self.ptr();
        for &i in axes.iter() {
            new_strides[i] = -new_strides[i];
            ptr.offset(self.strides()[i]);
        }
        if self.parent.is_none() {
            todo!()
        } else {
            todo!()
        }
    }

    fn fliplr(&self) -> Result<Self> {
        if self.ndim() < 2 {
            return Err(anyhow::Error::msg("_Tensor must have at least 2 dimensions for fliplr"));
        }
        self.flip(1)
    }

    fn flipud(&self) -> Result<Self> {
        if self.ndim() < 1 {
            return Err(anyhow::Error::msg("_Tensor must have at least 1 dimensions for flipud"));
        }
        self.flip(0)
    }

    fn tile<S: Into<Axis>>(&self, repeats: S) -> Result<Self> {
        let repeats: Vec<usize> = process_axes(repeats, self.ndim())?;
        let repeats: Vec<i64> = repeats
            .into_iter()
            .map(|x| x as i64)
            .collect::<Vec<i64>>();
        let final_repeats;
        let mut final_shape;
        if repeats.len() > self.ndim() {
            final_shape = try_pad_shape(self.shape().as_ref(), repeats.len());
            final_repeats = repeats.clone();
        } else {
            final_shape = self.shape().to_vec();
            final_repeats = try_pad_shape(repeats.as_ref(), self.ndim());
        }
        let mut res = self.reshape(&final_shape)?;
        let mut cnt = 0;
        for (idx, &i) in final_repeats.iter().enumerate() {
            if i == 1 {
                continue;
            } else {
                let tmp_shape = yield_one_before(res.shape().as_ref(), idx);
                res = res.reshape(tmp_shape)?;
                res = res.repeat(i as usize, (idx + cnt) as i16)?;
                final_shape[idx] *= i;
                cnt += 1;
            }
        }
        res.reshape(final_shape)
    }

    fn trim_zeros(&self, trim: &str) -> Result<Self> where Self::Meta: PartialEq {
        if !(trim == "fb" || trim == "f" || trim == "b") {
            return Err(anyhow::Error::msg("trim must be one of 'fb', 'f', 'b'"));
        }
        if self.ndim() > 1 {
            return Err(
                anyhow::Error::msg("_Tensor must have at most 1 dimension for trim_zeros method")
            );
        }
        let stride = self.strides()[0] as isize;
        let raw = self.as_raw();
        let mut ptr = raw.as_ptr();
        let mut left_len = 0;
        if trim.contains('f') {
            unsafe {
                for i in 0..raw.len() as isize {
                    if *ptr.offset(i * stride) != T::ZERO {
                        break;
                    } else {
                        left_len += 1;
                    }
                }
            }
        }
        let mut right_len = raw.len() as i64;
        if trim.contains('b') {
            unsafe {
                ptr = raw.as_ptr().offset(((raw.len() - 1) as isize) * stride);
                let stride = -stride;
                for i in 0..raw.len() as isize {
                    if *ptr.offset(i * stride) != T::ZERO {
                        break;
                    } else {
                        right_len -= 1;
                    }
                }
            }
        }
        todo!()
    }

    fn repeat(&self, repeats: usize, axes: i16) -> Result<_Tensor<T, Wgpu>> {
        let mut val: usize = axes as usize;
        if axes < 0 {
            val = self.shape().len() + (axes as usize);
        }
        let mut new_shape = yield_one_after(&self.shape(), val);
        let mut new_tensor: _Tensor<T, Wgpu> = self.reshape(&new_shape)?;
        new_shape[val + 1] *= repeats as i64;
        new_tensor = new_tensor.expand(new_shape)?;
        new_shape = self.shape().to_vec();
        new_shape[val] *= repeats as i64;
        Ok(new_tensor.contiguous()?.reshape(new_shape)?)
    }

    fn split(&self, indices_or_sections: &[i64], axis: i64) -> Result<Vec<Self>> {
        let mut new_axis = axis;
        if axis < 0 {
            new_axis = (self.ndim() as i64) + axis;
        }
        assert!(new_axis >= 0);
        todo!()
    }

    fn dsplit(&self, indices: &[i64]) -> Result<Vec<Self>> {
        if self.shape().len() < 3 {
            return Err(
                anyhow::Error::msg("_Tensor must have at least 3 dimensions for dsplit method")
            );
        }
        self.split(indices, 2)
    }

    fn hsplit(&self, indices: &[i64]) -> Result<Vec<Self>> {
        if self.shape().len() < 2 {
            return Err(
                anyhow::Error::msg("_Tensor must have at least 2 dimensions for hsplit method")
            );
        }
        self.split(indices, 1)
    }

    fn vsplit(&self, indices: &[i64]) -> Result<Vec<Self>> {
        if self.shape().len() < 1 {
            return Err(
                anyhow::Error::msg("_Tensor must have at least 1 dimensions for vsplit method")
            );
        }
        self.split(indices, 0)
    }

    fn swap_axes(&self, mut axis1: i64, mut axis2: i64) -> Result<Self> {
        ErrHandler::check_index_in_range(self.ndim(), &mut axis1)?;
        ErrHandler::check_index_in_range(self.ndim(), &mut axis2)?;
        let mut new_shape = self.shape().to_vec();
        let mut new_strides = self.strides().to_vec();
        new_shape.swap(axis1 as usize, axis2 as usize);
        new_strides.swap(axis1 as usize, axis2 as usize);
        let layout = Layout::new(new_shape, new_strides);
        todo!()
    }

    fn flatten<A>(&self, axis: A) -> Result<Self> where A: Into<Option<usize>> {
        let axis = axis.into().unwrap_or(1);
        let mut new_shape = vec![];
        let mut new_strides = vec![];
        let mut acc = 1;
        for (idx, (dim, stride)) in self.layout
            .shape()
            .iter()
            .zip(self.layout.strides().iter())
            .enumerate() {
            if idx == axis {
                acc *= dim;
                new_shape.push(acc);
                new_strides.push(*stride);
            } else if idx < axis {
                acc *= dim;
            } else {
                new_shape.push(*dim);
                new_strides.push(*stride);
            }
        }
        let layout = Layout::new(new_shape, new_strides);
        todo!()
    }
    
    fn permute_inv<A: Into<Axis>>(&self, axes: A) -> Result<Self> {
        todo!()
    }
}

impl<T> Random
    for Tensor<T, Wgpu>
    where
        T: CommonBounds + SampleUniform + num::Float + rand_distr::num_traits::FloatConst,
        <T as SampleUniform>::Sampler: Sync,
        StandardNormal: Distribution<T>,
        Open01: Distribution<T>,
        Exp1: Distribution<T>,
        OpenClosed01: Distribution<T>,
        Standard: Distribution<T>
{
    type Meta = T;

    fn randn<S: Into<Shape>>(shape: S) -> Result<Self> {
        todo!()
    }

    fn randn_like(&self) -> Result<Self> {
        todo!()
    }

    fn rand<S: Into<Shape>>(shape: S, low: Self::Meta, high: Self::Meta) -> Result<Self> {
        todo!()
    }

    fn rand_like(&self) -> Result<Self> {
        todo!()
    }

    fn beta<S: Into<Shape>>(a: Self::Meta, b: Self::Meta, shape: S) -> Result<Self> {
        todo!()
    }

    fn beta_like(&self, a: Self::Meta, b: Self::Meta) -> Result<Self> {
        todo!()
    }

    fn chisquare<S: Into<Shape>>(df: Self::Meta, shape: S) -> Result<Self> {
        todo!()
    }

    fn chisquare_like(&self, df: Self::Meta) -> Result<Self> {
        todo!()
    }

    fn exponential<S: Into<Shape>>(lambda: Self::Meta, shape: S) -> Result<Self> {
        todo!()
    }

    fn exponential_like(&self, lambda: Self::Meta) -> Result<Self> {
        todo!()
    }

    fn gamma<S: Into<Shape>>(gamm_shape: Self::Meta, scale: Self::Meta, shape: S) -> Result<Self> {
        todo!()
    }

    fn gamma_like(&self, shape: Self::Meta, scale: Self::Meta) -> Result<Self> {
        todo!()
    }

    fn gumbel<S: Into<Shape>>(mu: Self::Meta, beta: Self::Meta, shape: S) -> Result<Self> {
        todo!()
    }

    fn gumbel_like(&self, mu: Self::Meta, beta: Self::Meta) -> Result<Self> {
        todo!()
    }

    fn lognormal<S: Into<Shape>>(mean: Self::Meta, std: Self::Meta, shape: S) -> Result<Self> {
        todo!()
    }

    fn lognormal_like(&self, mean: Self::Meta, std: Self::Meta) -> Result<Self> {
        todo!()
    }

    fn normal_gaussian<S: Into<Shape>>(
        mean: Self::Meta,
        std: Self::Meta,
        shape: S
    ) -> Result<Self> {
        todo!()
    }

    fn normal_gaussian_like(&self, mean: Self::Meta, std: Self::Meta) -> Result<Self> {
        todo!()
    }

    fn pareto<S: Into<Shape>>(pareto_shape: Self::Meta, a: Self::Meta, shape: S) -> Result<Self> {
        todo!()
    }

    fn pareto_like(&self, pareto_shape: Self::Meta, a: Self::Meta) -> Result<Self> {
        todo!()
    }

    fn poisson<S: Into<Shape>>(lambda: Self::Meta, shape: S) -> Result<Self> {
        todo!()
    }

    fn poisson_like(&self, lambda: Self::Meta) -> Result<Self> {
        todo!()
    }

    fn weibull<S: Into<Shape>>(a: Self::Meta, b: Self::Meta, shape: S) -> Result<Self> {
        todo!()
    }

    fn weibull_like(&self, a: Self::Meta, b: Self::Meta) -> Result<Self> {
        todo!()
    }

    fn zipf<S: Into<Shape>>(n: u64, a: Self::Meta, shape: S) -> Result<Self> {
        todo!()
    }

    fn zipf_like(&self, n: u64, a: Self::Meta) -> Result<Self> {
        todo!()
    }

    fn triangular<S: Into<Shape>>(
        low: Self::Meta,
        high: Self::Meta,
        mode: Self::Meta,
        shape: S
    ) -> Result<Self> {
        todo!()
    }

    fn triangular_like(&self, low: Self::Meta, high: Self::Meta, mode: Self::Meta) -> Result<Self> {
        todo!()
    }

    fn bernoulli<S: Into<Shape>>(shape: S, p: Self::Meta) -> Result<Self>
        where Self::Meta: IntoScalar<f64>, bool: IntoScalar<Self::Meta>
    {
        todo!()
    }
}

impl<T> Display for _Tensor<T, Wgpu> where T: CommonBounds + bytemuck::Pod + Debug {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_cpu())
    }
}

impl<T> Debug for _Tensor<T, Wgpu> where T: CommonBounds + bytemuck::Pod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let res_size = self.size() * std::mem::size_of::<T>();
        let read_buffer = self.device().create_buffer(
            &(wgpu::BufferDescriptor {
                label: Some("Result Buffer"),
                size: res_size as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        );
        let mut encoder = self
            .device()
            .create_command_encoder(&(wgpu::CommandEncoderDescriptor { label: None }));
        encoder.copy_buffer_to_buffer(self.buffer(), 0, &read_buffer, 0, res_size as u64);
        self.device().queue.submit(Some(encoder.finish()));
        let buffer_slice = read_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device().poll(wgpu::Maintain::wait()).panic_on_timeout();
        let tensor = _Tensor::<T, Cpu>::empty(self.shape()).unwrap();
        pollster::block_on(async {
            if let Ok(Ok(())) = receiver.recv_async().await {
                let data = buffer_slice.get_mapped_range();
                let result: &[T] = bytemuck::cast_slice(&data);
                tensor
                    .as_raw_mut()
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(i, x)| {
                        *x = result[i];
                    });
            }
        });
        write!(f, "{}", tensor)
    }
}

impl<T> Into<Tensor<T, Wgpu>> for _Tensor<T, Wgpu> {
    fn into(self) -> Tensor<T, Wgpu> {
        todo!()
    }
}

impl<T> Into<_Tensor<T, Wgpu>> for &_Tensor<T, Wgpu> where T: CommonBounds {
    fn into(self) -> _Tensor<T, Wgpu> {
        todo!()
    }
}

impl<T> Into<Tensor<T, Wgpu>> for &Tensor<T, Wgpu> {
    fn into(self) -> Tensor<T, Wgpu> {
        todo!()
    }
}

impl<'a, T> Into<_Tensor<T, Wgpu>> for &'a [T] {
    fn into(self) -> _Tensor<T, Wgpu> {
        let shape = vec![self.len() as i64];
        let strides = vec![1];
        let layout = Layout::new(shape, strides);
        let mem_layout = std::alloc::Layout
            ::from_size_align(self.len() * std::mem::size_of::<T>(), ALIGN)
            .unwrap();
        let ptr = unsafe { CACHE.allocate(mem_layout.clone()) };
        unsafe {
            std::ptr::copy_nonoverlapping(self.as_ptr(), ptr as *mut T, self.len());
        }
        todo!()
    }
}

impl<T, B> Drop for _Tensor<T, B> where B: BackendTy + BackendDevice {
    fn drop(&mut self) {
        match B::ID {
            0 => {
                unsafe {
                    CACHE.deallocate(self._backend._backend.ptr() as *mut u8, &self.mem_layout)
                }
            }
            2 => {
                unsafe {
                    WGPU_CACHE.deallocate(
                        self._backend._backend.wgpu_device(),
                        self._backend._backend.buffer(),
                        &self.mem_layout
                    )
                }
            }
            _ => { panic!("Invalid Backend ID") }
        }
    }
}
