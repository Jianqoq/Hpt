use std::{borrow::Cow, panic::Location};
use wgpu::util::DeviceExt;
use tensor_common::{ shape_utils::try_pad_shape, strides_utils::preprocess_strides };
use tensor_traits::{ CommonBounds, TensorInfo };
use tensor_types::{ dtype::TypeCommon, type_promote::NormalOut };

use crate::{ backend::Wgpu, tensor_base::_Tensor };

pub(crate) fn binop<A, B>(
    op: &str,
    a: &_Tensor<A, Wgpu>,
    b: &_Tensor<B, Wgpu>,
    location: &'static Location<'static>
)
    -> _Tensor<<A as NormalOut<B>>::Output, Wgpu>
    where
        A: CommonBounds + NormalOut<B> + bytemuck::Pod + TypeCommon,
        B: CommonBounds + bytemuck::Pod + TypeCommon,
        <A as NormalOut<B>>::Output: CommonBounds + bytemuck::Pod + TypeCommon
{
    let grp_size_x = 32;
    let grp_size_y = 1;
    let num_grp_x = 512;
    let num_grp_y = 512;

    let res_shape = a.layout().broadcast(&b.layout(), location).expect("Failed to broadcast shapes");

    let a_strides: Vec<i64> = preprocess_strides(
        &try_pad_shape(a.shape(), res_shape.ndim()),
        a.strides()
    );
    let b_strides: Vec<i64> = preprocess_strides(
        &try_pad_shape(b.shape(), res_shape.ndim()),
        b.strides()
    );

    let res = _Tensor::<<A as NormalOut<B>>::Output, Wgpu>
        ::empty(res_shape.shape(), a.device())
        .expect("Failed to create tensor");

    let outer = res_shape.size() / res_shape.shape().last().unwrap();
    let inner = *res_shape.shape().last().unwrap();

    let kernel = include_str!("../../wgpu_kernels/binary.wgsl")
        .replace("a_ty", A::STR)
        .replace("b_ty", B::STR)
        .replace("c_ty", <A as NormalOut<B>>::Output::STR)
        .replace("op_place_holder", op)
        .replace("GRP_SIZE_X", &grp_size_x.to_string())
        .replace("GRP_SIZE_Y", &grp_size_y.to_string())
        .replace("NUM_GRP_X", &num_grp_x.to_string())
        .replace("NUM_GRP_Y", &num_grp_y.to_string())
        .replace("outer_loop_size", &outer.to_string())
        .replace("inner_loop_size", &inner.to_string())
        .replace("a_last_stride", &a_strides.last().unwrap().to_string())
        .replace("b_last_stride", &b_strides.last().unwrap().to_string())
        .replace("c_last_stride", &res.strides().last().unwrap().to_string())
        .replace("res_ndim", &res.ndim().to_string());
    let device = a.device().clone();
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&kernel)),
    });

    let res_buffer = res.buffer();

    let res_strides_buffer = device.create_buffer_init(
        &(wgpu::util::BufferInitDescriptor {
            label: Some("res_strides_buffer"),
            contents: bytemuck::cast_slice(res.strides()),
            usage: wgpu::BufferUsages::STORAGE |
            wgpu::BufferUsages::COPY_DST |
            wgpu::BufferUsages::COPY_SRC,
        })
    );

    let res_shape_buffer = device.create_buffer_init(
        &(wgpu::util::BufferInitDescriptor {
            label: Some("res_shape_buffer"),
            contents: bytemuck::cast_slice(res.shape()),
            usage: wgpu::BufferUsages::STORAGE |
            wgpu::BufferUsages::COPY_DST |
            wgpu::BufferUsages::COPY_SRC,
        })
    );

    let a_buffer = a.buffer();

    let a_strides_buffer = device.create_buffer_init(
        &(wgpu::util::BufferInitDescriptor {
            label: Some("a_strides_buffer"),
            contents: bytemuck::cast_slice(&a_strides),
            usage: wgpu::BufferUsages::STORAGE |
            wgpu::BufferUsages::COPY_DST |
            wgpu::BufferUsages::COPY_SRC,
        })
    );

    let b_buffer = b.buffer();

    let b_strides_buffer = device.create_buffer_init(
        &(wgpu::util::BufferInitDescriptor {
            label: Some("b_strides_buffer"),
            contents: bytemuck::cast_slice(&b_strides),
            usage: wgpu::BufferUsages::STORAGE |
            wgpu::BufferUsages::COPY_DST |
            wgpu::BufferUsages::COPY_SRC,
        })
    );

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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                    resource: a_strides_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_strides_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: res_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: res_strides_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: res_shape_buffer.as_entire_binding(),
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

    let queue = device.queue();
    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));

    res
}
