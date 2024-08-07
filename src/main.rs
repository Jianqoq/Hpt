use std::{ borrow::Cow, fmt::Debug };
use tensor_common::shape_utils::try_pad_shape;
use tensor_common::strides_utils::preprocess_strides;
use tensor_dyn::backend::{ Cpu, Wgpu };
use tensor_dyn::ops::wgpu::buffer_helper::WgpuDevice;
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::{ tensor::Tensor, CommonBounds, TensorCreator };
use tensor_types::dtype::TypeCommon;
use tensor_types::type_promote::NormalOut;
use wgpu::util::DeviceExt;
use tensor_dyn::TensorInfo;

async fn binop<A, B>(
    kernel: &str,
    a: &_Tensor<A, Wgpu>,
    b: &_Tensor<B, Wgpu>
)
    -> _Tensor<<A as NormalOut<B>>::Output, Wgpu>
    where
        A: CommonBounds + NormalOut<B> + bytemuck::Pod + TypeCommon,
        B: CommonBounds + bytemuck::Pod + TypeCommon,
        <A as NormalOut<B>>::Output: CommonBounds + bytemuck::Pod + Debug + TypeCommon
{
    let grp_size_x = 16;
    let grp_size_y = 16;
    let num_grp_x = 1024;
    let num_grp_y = 1024;

    let res_shape = a.layout().broadcast(&b.layout()).expect("Failed to broadcast shapes");

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

    let kernel = kernel
        .replace("GRP_SIZE_X", &grp_size_x.to_string())
        .replace("GRP_SIZE_Y", &grp_size_y.to_string())
        .replace("NUM_GRP_X", &num_grp_x.to_string())
        .replace("NUM_GRP_Y", &num_grp_y.to_string())
        .replace("a_ty", &A::ID.to_string())
        .replace("b_ty", &B::ID.to_string())
        .replace("c_ty", &<A as NormalOut<B>>::Output::ID.to_string())
        .replace("outer_loop_size", &outer.to_string())
        .replace("inner_loop_size", &inner.to_string())
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

    let (sender2, receiver2) = flume::bounded(1);
    time_slice.map_async(wgpu::MapMode::Read, move |v| sender2.send(v).unwrap());

    device.poll(wgpu::Maintain::wait()).panic_on_timeout();

    if let Ok(Ok(())) = receiver2.recv_async().await {
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

use tensor_dyn::ShapeManipulate;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let device = WgpuDevice::new(
        wgpu::Backends::VULKAN | wgpu::Backends::DX12,
        wgpu::Features::SHADER_INT64 |
            wgpu::Features::BUFFER_BINDING_ARRAY |
            wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY |
            wgpu::Features::TIMESTAMP_QUERY |
            wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
    );
    let a = _Tensor::<f32, Wgpu>
        ::arange(0, 1024 * 1024 * 128, &device).await?
        .reshape(&[1024, 1024, 128])?;
    let b = _Tensor::<f32, Wgpu>
        ::arange(0, 1024 * 1024 * 128, &device).await?
        .reshape(&[1024, 1024, 128])?;
    let res = binop(include_str!("shader.wgsl"), &a, &b).await;

    println!("{}", res);
    Ok(())
}
