use std::{ borrow::Cow, fmt::Debug };
use tensor_common::shape_utils::try_pad_shape;
use tensor_common::strides_utils::preprocess_strides;
use tensor_dyn::{ tensor::Tensor, CommonBounds, TensorCreator };
use tensor_types::dtype::TypeCommon;
use tensor_types::type_promote::NormalOut;
use wgpu::util::DeviceExt;
use tensor_dyn::TensorInfo;

async fn create_device() -> (wgpu::Device, wgpu::Queue) {
    // Instantiates instance of WebGPU
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::DX12,
        flags: InstanceFlags::VALIDATION,
        dx12_shader_compiler: Dx12Compiler::Dxc { dxil_path: None, dxc_path: None },
        gles_minor_version: Gles3MinorVersion::default(),
    });

    // `request_adapter` instantiates the general connection to the GPU
    let adapter = instance
        .request_adapter(
            &(RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
        ).await
        .unwrap();

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    let limits = wgpu::Limits {
        max_buffer_size: 20 * 1024 * 1024 * 1024,
        max_storage_buffers_per_shader_stage: 12,
        ..wgpu::Limits::default()
    };
    adapter
        .request_device(
            &(wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::SHADER_INT64,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            }),
            None
        ).await
        .unwrap()
}

async fn binop<A, B>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    kernel: &str,
    a: &Tensor<A>,
    b: &Tensor<B>
)
    -> Tensor<<A as NormalOut<B>>::Output>
    where
        A: CommonBounds + NormalOut<B> + bytemuck::Pod + TypeCommon,
        B: CommonBounds + bytemuck::Pod + TypeCommon,
        <A as NormalOut<B>>::Output: CommonBounds + bytemuck::Pod + Debug + TypeCommon
{
    let res_shape = a.layout().broadcast(&b.layout()).expect("Failed to broadcast shapes");

    let a_strides: Vec<i64> = preprocess_strides(
        &try_pad_shape(a.shape(), res_shape.ndim()),
        a.strides()
    );
    let b_strides: Vec<i64> = preprocess_strides(
        &try_pad_shape(b.shape(), res_shape.ndim()),
        b.strides()
    );

    let res = Tensor::<<A as NormalOut<B>>::Output>
        ::empty(res_shape.shape())
        .expect("Failed to create tensor");

    let kernel = kernel
        .replace("prg_place_holder", &(res.ndim() - 1).to_string())
        .replace("GRP_SIZE_X", &(16).to_string())
        .replace("GRP_SIZE_Y", &(16).to_string())
        .replace("NUM_GRP_X", &(64).to_string())
        .replace("NUM_GRP_Y", &(64).to_string())
        .replace("a_ty", &A::ID.to_string())
        .replace("b_ty", &B::ID.to_string())
        .replace("c_ty", &<A as NormalOut<B>>::Output::ID.to_string());
    println!("{}", kernel);
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&kernel)),
    });

    // Gets the size in bytes of the buffer.
    let res_size =
        (res.size() as u64) *
        (std::mem::size_of::<<A as NormalOut<B>>::Output>() as wgpu::BufferAddress);

    // Instantiates buffer without data.
    // `usage` of buffer specifies how it can be used:
    //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
    //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
    let res_buffer = device.create_buffer(
        &(wgpu::BufferDescriptor {
            label: Some("res_buffer"),
            size: res_size,
            usage: wgpu::BufferUsages::STORAGE |
            wgpu::BufferUsages::COPY_SRC |
            wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    );

    let result_buffer = device.create_buffer(
        &(wgpu::BufferDescriptor {
            label: Some("Result Buffer"),
            size: res_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    );

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

    let res_ndim_buffer = device.create_buffer_init(
        &(wgpu::util::BufferInitDescriptor {
            label: Some("res_ndim_buffer"),
            contents: bytemuck::cast_slice(&[res_shape.ndim()]),
            usage: wgpu::BufferUsages::STORAGE |
            wgpu::BufferUsages::COPY_DST |
            wgpu::BufferUsages::COPY_SRC,
        })
    );

    let outer_loop_size = res_shape.size() / res_shape.shape().last().unwrap();
    let inner_loop_size = *res_shape.shape().last().unwrap();

    let outer_loop_size_buffer = device.create_buffer_init(
        &(wgpu::util::BufferInitDescriptor {
            label: Some("outer_loop_size_buffer"),
            contents: bytemuck::cast_slice(&[outer_loop_size]),
            usage: wgpu::BufferUsages::STORAGE |
            wgpu::BufferUsages::COPY_DST |
            wgpu::BufferUsages::COPY_SRC,
        })
    );

    let inner_loop_size_buffer = device.create_buffer_init(
        &(wgpu::util::BufferInitDescriptor {
            label: Some("inner_loop_size_buffer"),
            contents: bytemuck::cast_slice(&[inner_loop_size]),
            usage: wgpu::BufferUsages::STORAGE |
            wgpu::BufferUsages::COPY_DST |
            wgpu::BufferUsages::COPY_SRC,
        })
    );

    // Instantiates buffer with data (`numbers`).
    // Usage allowing the buffer to be:
    //   A storage buffer (can be bound within a bind group and thus available to a shader).
    //   The destination of a copy.
    //   The source of a copy.
    let a_buffer = device.create_buffer_init(
        &(wgpu::util::BufferInitDescriptor {
            label: Some("a buffer"),
            contents: bytemuck::cast_slice(a.as_raw()),
            usage: wgpu::BufferUsages::STORAGE |
            wgpu::BufferUsages::COPY_DST |
            wgpu::BufferUsages::COPY_SRC,
        })
    );

    let a_strides_buffer = device.create_buffer_init(
        &(wgpu::util::BufferInitDescriptor {
            label: Some("a_strides_buffer"),
            contents: bytemuck::cast_slice(&a_strides),
            usage: wgpu::BufferUsages::STORAGE |
            wgpu::BufferUsages::COPY_DST |
            wgpu::BufferUsages::COPY_SRC,
        })
    );

    let b_buffer = device.create_buffer_init(
        &(wgpu::util::BufferInitDescriptor {
            label: Some("b_buffer"),
            contents: bytemuck::cast_slice(b.as_raw()),
            usage: wgpu::BufferUsages::STORAGE |
            wgpu::BufferUsages::COPY_DST |
            wgpu::BufferUsages::COPY_SRC,
        })
    );

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
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
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
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: outer_loop_size_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: inner_loop_size_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: res_ndim_buffer.as_entire_binding(),
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

    // A bind group defines how buffers are accessed by shaders.
    // It is to WebGPU what a descriptor set is to Vulkan.
    // `binding` here refers to the `binding` of a buffer in the shader (`layout(set = 0, binding = 0) buffer`).

    // A pipeline specifies the operation of a shader

    // Instantiates the pipeline.
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

    // A command encoder executes one or many pipelines.
    // It is to WebGPU what a command buffer is to Vulkan.
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

        cpass.dispatch_workgroups(64, 64, 1); // Number of cells to run, the (x,y,z) size of item being processed
    }
    let now = std::time::Instant::now();
    encoder.copy_buffer_to_buffer(&res_buffer, 0, &result_buffer, 0, res_size);
    println!("copy time: {:?}", now.elapsed());
    let now = std::time::Instant::now();
    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));

    // Note that we're not calling `.await` here.
    let buffer_slice = result_buffer.slice(..);
    // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    device.poll(wgpu::Maintain::wait()).panic_on_timeout();

    // Awaits until `buffer_future` can be read from
    if let Ok(Ok(())) = receiver.recv_async().await {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Since contents are got in bytes, this converts these bytes back to u32
        let result: &[<A as NormalOut<B>>::Output] = bytemuck::cast_slice(&data);
        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        // If you are familiar with C++ these 2 lines can be thought of similarly to:
        //   delete myPointer;
        //   myPointer = NULL;
        // It effectively frees the memory

        // Returns data from buffer
        res.as_raw_mut().copy_from_slice(result);
        drop(data);
        result_buffer.unmap(); // Unmaps buffer from memory
        println!("Time taken: {:?}", now.elapsed());
        res
    } else {
        panic!("failed to run compute on gpu!")
    }
}
use tensor_dyn::ShapeManipulate;
use wgpu::{ Dx12Compiler, Gles3MinorVersion, InstanceFlags, RequestAdapterOptions };
fn main() -> anyhow::Result<()> {
    println!(
        "allocating memory on gpu {}",
        (std::mem::size_of::<f32>() * 1024 * 1024 * 16) / 1024 / 1024
    );
    let a = Tensor::<f32>
        ::arange(0, 1024 * 1024 * 20)
        .unwrap()
        .reshape(&[1024, 1024, 20])
        .unwrap();
    let b = Tensor::<f32>
        ::arange(0, 1024 * 1024 * 20)
        .unwrap()
        .reshape(&[1024, 1024, 20])
        .unwrap();
    {
        pollster::block_on(async {
            let now = std::time::Instant::now();
            let (device, queue) = create_device().await;
            println!("get device time: {:?}", now.elapsed());
            let res = binop(&device, &queue, include_str!("shader.wgsl"), &a, &b).await;
        });
    }

    let now = std::time::Instant::now();
    {
        let res = a + b;
    }
    println!("Time taken: {:?}", now.elapsed());
    Ok(())
}
