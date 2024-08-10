use std::borrow::Cow;

use bytemuck::Pod;
use tensor_traits::{ CommonBounds, FloatUaryOps, TensorInfo };
use tensor_types::{ dtype::TypeCommon, type_promote::FloatOut };

use crate::{ backend::Wgpu, ops::cpu::unary::FloatType, tensor_base::_Tensor };

pub(crate) async fn unary<A>(op: &str, a: &_Tensor<A, Wgpu>) -> _Tensor<<A as FloatOut>::Output, Wgpu>
    where
        A: CommonBounds + FloatOut + bytemuck::Pod + TypeCommon,
        <A as FloatOut>::Output: CommonBounds + bytemuck::Pod + TypeCommon
{
    let grp_size_x = 16;
    let grp_size_y = 16;
    let num_grp_x = 1024;
    let num_grp_y = 1024;

    let res = _Tensor::<<A as FloatOut>::Output, Wgpu>
        ::empty(a.shape(), a.device())
        .expect("Failed to create tensor");

    let size = a.size();

    let kernel = include_str!("../../wgpu_kernels/unary.wgsl")
        .replace("a_ty", A::STR)
        .replace("c_ty", <A as FloatOut>::Output::STR)
        .replace("op_place_holder", op)
        .replace("GRP_SIZE_X", &grp_size_x.to_string())
        .replace("GRP_SIZE_Y", &grp_size_y.to_string())
        .replace("NUM_GRP_X", &num_grp_x.to_string())
        .replace("NUM_GRP_Y", &num_grp_y.to_string())
        .replace("TOTAL_SIZE", &size.to_string());

    let device = a.device().clone();
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&kernel)),
    });

    let res_buffer = res.buffer();

    let a_buffer = a.buffer();

    // Instantiates the bind group, once again specifying the binding of buffers.
    let bind_group_layout = device.create_bind_group_layout(
        &(wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    );

    let bind_group = device.create_bind_group(
        &(wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: res_buffer.as_entire_binding(),
                },
            ],
        })
    );

    let pipeline_layout = device.create_pipeline_layout(
        &(wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        })
    );

    let compute_pipeline = device.create_compute_pipeline(
        &(wgpu::ComputePipelineDescriptor {
            label: Some("compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &cs_module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        })
    );

    let query_set = device.create_query_set(
        &(wgpu::QuerySetDescriptor {
            label: Some("Timestamp Query Set"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        })
    );

    let mut encoder = device.create_command_encoder(
        &(wgpu::CommandEncoderDescriptor { label: None })
    );

    encoder.write_timestamp(&query_set, 0);
    {
        let mut cpass = encoder.begin_compute_pass(
            &(wgpu::ComputePassDescriptor {
                label: Some("compute_pass"),
                timestamp_writes: None,
            })
        );
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.insert_debug_marker("compute collatz iterations");

        cpass.dispatch_workgroups(num_grp_x, num_grp_y, 1); // Number of cells to run, the (x,y,z) size of item being processed
    }
    encoder.write_timestamp(&query_set, 1);

    let timestamp_buffer = device.create_buffer(
        &(wgpu::BufferDescriptor {
            label: Some("Timestamp Buffer"),
            size: (std::mem::size_of::<u64>() as u64) * 2,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
            mapped_at_creation: false,
        })
    );
    let readback_buffer = device.create_buffer(
        &(wgpu::BufferDescriptor {
            label: Some("Timestamp Buffer"),
            size: (std::mem::size_of::<u64>() as u64) * 2,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    );
    encoder.resolve_query_set(&query_set, 0..2, &timestamp_buffer, 0);
    encoder.copy_buffer_to_buffer(
        &timestamp_buffer,
        0,
        &readback_buffer,
        0,
        (std::mem::size_of::<u64>() as u64) * 2
    );

    let queue = device.queue();
    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));

    let time_slice = readback_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    time_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    device.poll(wgpu::Maintain::wait()).panic_on_timeout();

    if let Ok(Ok(())) = receiver.recv_async().await {
        let data = time_slice.get_mapped_range();
        let time: &[u64] = bytemuck::cast_slice(&data);
        let start_time = time[0];
        let end_time = time[1];
        let duration_ns = end_time - start_time;
        let duration_ms = (duration_ns as f64) / 1_000_000.0;
        println!("Kernel execution time: {:.3} ms", duration_ms);
    } else {
        panic!("failed to run compute on gpu!");
    }

    res
}

impl<T> _Tensor<T, Wgpu>
    where T: FloatOut + CommonBounds + Pod, FloatType<T>: CommonBounds + Pod
{

    pub async fn sin(&self) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        Ok(unary("sin", self).await)
    }

    pub async fn cos(&self) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        Ok(unary("cos", self).await)
    }

    pub async fn tan(&self) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        Ok(unary("tan", self).await)
    }

    pub async fn asin(&self) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        Ok(unary("asin", self).await)
    }

    pub async fn acos(&self) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        Ok(unary("acos", self).await)
    }

    pub async fn atan(&self) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        Ok(unary("atan", self).await)
    }

    pub async fn sinh(&self) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        Ok(unary("sinh", self).await)
    }

    pub async fn cosh(&self) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        Ok(unary("cosh", self).await)
    }

    pub async fn tanh(&self) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        Ok(unary("tanh", self).await)
    }

    pub async fn asinh(&self) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        Ok(unary("asinh", self).await)
    }

    pub async fn acosh(&self) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        Ok(unary("acosh", self).await)
    }

    pub async fn atanh(&self) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        Ok(unary("atanh", self).await)
    }

    pub async fn sin_<U>(&self, _: U) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }

    pub async fn cos_<U>(&self, _: U) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }

    pub async fn tan_<U>(&self, _: U) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }

    pub async fn asin_<U>(&self, _: U) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }

    pub async fn acos_<U>(&self, _: U) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }

    pub async fn atan_<U>(&self, _: U) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }

    pub async fn sinh_<U>(&self, _: U) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }

    pub async fn cosh_<U>(&self, _: U) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }

    pub async fn tanh_<U>(&self, _: U) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }

    pub async fn asinh_<U>(&self, _: U) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }

    pub async fn acosh_<U>(&self, _: U) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }

    pub async fn atanh_<U>(&self, _: U) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }

    pub async fn exp(&self) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        Ok(unary("exp", self).await)
    }

    pub async fn exp_<U>(&self, _: U) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }

    pub async fn exp2(&self) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        Ok(unary("exp2", self).await)
    }

    pub async fn exp2_<U>(&self, _: U) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }

    pub async fn sqrt(&self) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        Ok(unary("sqrt", self).await)
    }

    pub async fn sqrt_<U>(&self, _: U) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }

    pub async fn recip(&self) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        Ok(unary(&format!("{}(1) / ", T::STR), self).await)
    }

    pub async fn recip_<U>(&self, _: U) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }

    pub async fn ln(&self) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        Ok(unary("ln", self).await)
    }

    pub async fn ln_<U>(&self, _: U) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }

    pub async fn log2(&self) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        Ok(unary("log2", self).await)
    }

    pub async fn log2_<U>(&self, _: U) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }

    pub async fn log10(&self) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        todo!()
    }

    pub async fn log10_<U>(&self, _: U) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }

    pub async fn celu(&self, _: FloatType<T>) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        todo!()
    }

    pub async fn celu_<U>(&self, _: FloatType<T>, _: U) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }

    pub async fn sigmoid(&self) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        todo!()
    }

    pub async fn sigmoid_<U>(&self, _: U) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }

    pub async fn elu(&self, _: FloatType<T>) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        todo!()
    }

    pub async fn elu_<U>(&self, _: FloatType<T>, _: U) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }

    pub async fn leaky_relu(&self, _: FloatType<T>) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        todo!()
    }

    pub async fn leaky_relu_<U>(&self, _: FloatType<T>, _: U) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }

    pub async fn gelu(&self) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        todo!()
    }

    pub async fn gelu_<U>(&self, _: U) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }

    pub async fn selu(
        &self,
        _: Option<FloatType<T>>,
        _: Option<FloatType<T>>
    ) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        todo!()
    }

    pub async fn selu_<U>(
        &self,
        _: Option<FloatType<T>>,
        _: Option<FloatType<T>>,
        _: U
    ) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }

    pub async fn hard_sigmoid(
        &self,
        _: Option<FloatType<T>>,
        _: Option<FloatType<T>>
    ) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        todo!()
    }

    pub async fn hard_sigmoid_<U>(
        &self,
        _: Option<FloatType<T>>,
        _: Option<FloatType<T>>,
        _: U
    ) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }

    pub async fn hard_swish(&self) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        todo!()
    }

    pub async fn hard_swish_<U>(&self, _: U) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }

    pub async fn relu6(&self) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        todo!()
    }

    pub async fn relu6_<U>(&self, _: U) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }

    pub async fn softplus(&self) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        todo!()
    }

    pub async fn softplus_<U>(&self, _: U) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }

    pub async fn softsign(&self) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        todo!()
    }

    pub async fn softsign_<U>(&self, _: U) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }

    pub async fn mish(&self) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>> {
        todo!()
    }

    pub async fn mish_<U>(&self, _: U) -> anyhow::Result<_Tensor<FloatType<T>, Wgpu>>
        where U: tensor_traits::BaseTensor<Output = _Tensor<FloatType<T>, Wgpu>>
    {
        todo!()
    }
}
