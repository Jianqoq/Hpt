use std::borrow::Cow;

use bytemuck::Pod;
use tensor_traits::{ CommonBounds, FloatUaryOps, TensorInfo };
use tensor_types::{ dtype::TypeCommon, type_promote::FloatOut };

use crate::{ backend::Wgpu, ops::cpu::unary::FloatType, tensor_base::_Tensor };

pub(crate) fn unary<A>(op: &str, a: &_Tensor<A, Wgpu>) -> _Tensor<<A as FloatOut>::Output, Wgpu>
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

    let mut encoder = device.create_command_encoder(
        &(wgpu::CommandEncoderDescriptor { label: None })
    );

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

    let queue = device.queue();
    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));

    res
}

impl<T> FloatUaryOps
    for _Tensor<T, Wgpu>
    where T: FloatOut + CommonBounds + Pod, FloatType<T>: CommonBounds + Pod
{
    type Output = _Tensor<FloatType<T>, Wgpu>;

    type InplaceOutput = _Tensor<FloatType<T>, Wgpu>;

    type OutputMeta = FloatType<T>;

    fn sin(&self) -> anyhow::Result<Self::Output> {
        Ok(unary("sin", self))
    }

    fn cos(&self) -> anyhow::Result<Self::Output> {
        Ok(unary("cos", self))
    }

    fn tan(&self) -> anyhow::Result<Self::Output> {
        Ok(unary("tan", self))
    }

    fn asin(&self) -> anyhow::Result<Self::Output> {
        Ok(unary("asin", self))
    }

    fn acos(&self) -> anyhow::Result<Self::Output> {
        Ok(unary("acos", self))
    }

    fn atan(&self) -> anyhow::Result<Self::Output> {
        Ok(unary("atan", self))
    }

    fn sinh(&self) -> anyhow::Result<Self::Output> {
        Ok(unary("sinh", self))
    }

    fn cosh(&self) -> anyhow::Result<Self::Output> {
        Ok(unary("cosh", self))
    }

    fn tanh(&self) -> anyhow::Result<Self::Output> {
        Ok(unary("tanh", self))
    }

    fn asinh(&self) -> anyhow::Result<Self::Output> {
        Ok(unary("asinh", self))
    }

    fn acosh(&self) -> anyhow::Result<Self::Output> {
        Ok(unary("acosh", self))
    }

    fn atanh(&self) -> anyhow::Result<Self::Output> {
        Ok(unary("atanh", self))
    }

    fn sin_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }

    fn cos_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }

    fn tan_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }

    fn asin_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }

    fn acos_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }

    fn atan_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }

    fn sinh_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }

    fn cosh_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }

    fn tanh_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }

    fn asinh_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }

    fn acosh_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }

    fn atanh_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }

    fn exp(&self) -> anyhow::Result<Self::Output> {
        Ok(unary("exp", self))
    }

    fn exp_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }

    fn exp2(&self) -> anyhow::Result<Self::Output> {
        Ok(unary("exp2", self))
    }

    fn exp2_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }

    fn sqrt(&self) -> anyhow::Result<Self::Output> {
        Ok(unary("sqrt", self))
    }

    fn sqrt_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }

    fn recip(&self) -> anyhow::Result<Self::Output> {
        Ok(unary(&format!("{}(1) / ", T::STR), self))
    }

    fn recip_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }

    fn ln(&self) -> anyhow::Result<Self::Output> {
        Ok(unary("ln", self))
    }

    fn ln_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }

    fn log2(&self) -> anyhow::Result<Self::Output> {
        Ok(unary("log2", self))
    }

    fn log2_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }

    fn log10(&self) -> anyhow::Result<Self::Output> {
        todo!()
    }

    fn log10_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }

    fn celu(&self, alpha: Self::OutputMeta) -> anyhow::Result<Self::Output> {
        todo!()
    }

    fn celu_<U>(&self, alpha: Self::OutputMeta, out: U) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }

    fn sigmoid(&self) -> anyhow::Result<Self::Output> {
        todo!()
    }

    fn sigmoid_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }

    fn elu(&self, alpha: Self::OutputMeta) -> anyhow::Result<Self::Output> {
        todo!()
    }

    fn elu_<U>(&self, alpha: Self::OutputMeta, out: U) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }

    fn leaky_relu(&self, alpha: Self::OutputMeta) -> anyhow::Result<Self::Output> {
        todo!()
    }

    fn leaky_relu_<U>(&self, alpha: Self::OutputMeta, out: U) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }

    fn gelu(&self) -> anyhow::Result<Self::Output> {
        todo!()
    }

    fn gelu_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }

    fn selu(
        &self,
        alpha: Option<Self::OutputMeta>,
        gamma: Option<Self::OutputMeta>
    ) -> anyhow::Result<Self::Output> {
        todo!()
    }

    fn selu_<U>(
        &self,
        alpha: Option<Self::OutputMeta>,
        gamma: Option<Self::OutputMeta>,
        out: U
    ) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }

    fn hard_sigmoid(
        &self,
        alpha: Option<Self::OutputMeta>,
        beta: Option<Self::OutputMeta>
    ) -> anyhow::Result<Self::Output> {
        todo!()
    }

    fn hard_sigmoid_<U>(
        &self,
        alpha: Option<Self::OutputMeta>,
        beta: Option<Self::OutputMeta>,
        out: U
    ) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }

    fn hard_swish(&self) -> anyhow::Result<Self::Output> {
        todo!()
    }

    fn hard_swish_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }

    fn relu6(&self) -> anyhow::Result<Self::Output> {
        todo!()
    }

    fn relu6_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }

    fn softplus(&self) -> anyhow::Result<Self::Output> {
        todo!()
    }

    fn softplus_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }

    fn softsign(&self) -> anyhow::Result<Self::Output> {
        todo!()
    }

    fn softsign_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }

    fn mish(&self) -> anyhow::Result<Self::Output> {
        todo!()
    }

    fn mish_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: tensor_traits::BaseTensor<Output = Self::InplaceOutput>
    {
        todo!()
    }
}
