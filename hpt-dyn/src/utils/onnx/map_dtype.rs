use hpt_types::dtype::DType;

use crate::onnx::tensor_proto;

pub(crate) fn to_dtype(ty: i32) -> DType {
    let data_type = tensor_proto::DataType::try_from(ty).expect("invalid data type");
    match data_type {
        tensor_proto::DataType::Undefined => unimplemented!("undefined data type is not supported"),
        tensor_proto::DataType::Float => DType::F32,
        tensor_proto::DataType::Uint8 => DType::U8,
        tensor_proto::DataType::Int8 => DType::I8,
        tensor_proto::DataType::Uint16 => DType::U16,
        tensor_proto::DataType::Int16 => DType::I16,
        tensor_proto::DataType::Int32 => DType::I32,
        tensor_proto::DataType::Int64 => DType::I64,
        tensor_proto::DataType::String => unimplemented!("string data type is not supported"),
        tensor_proto::DataType::Bool => DType::Bool,
        tensor_proto::DataType::Float16 => DType::F16,
        tensor_proto::DataType::Double => unimplemented!("double data type is not supported"),
        tensor_proto::DataType::Uint32 => DType::U32,
        tensor_proto::DataType::Uint64 => unimplemented!("uint64 data type is not supported"),
        tensor_proto::DataType::Complex64 => unimplemented!("complex64 data type is not supported"),
        tensor_proto::DataType::Complex128 => {
            unimplemented!("complex128 data type is not supported")
        }
        tensor_proto::DataType::Bfloat16 => DType::BF16,
        tensor_proto::DataType::Float8e4m3fn => {
            unimplemented!("float8e4m3fn data type is not supported")
        }
        tensor_proto::DataType::Float8e4m3fnuz => {
            unimplemented!("float8e4m3fnuz data type is not supported")
        }
        tensor_proto::DataType::Float8e5m2 => {
            unimplemented!("float8e5m2 data type is not supported")
        }
        tensor_proto::DataType::Float8e5m2fnuz => {
            unimplemented!("float8e5m2fnuz data type is not supported")
        }
        tensor_proto::DataType::Uint4 => unimplemented!("uint4 data type is not supported"),
        tensor_proto::DataType::Int4 => unimplemented!("int4 data type is not supported"),
        tensor_proto::DataType::Float4e2m1 => {
            unimplemented!("float4e2m1 data type is not supported")
        }
    }
}
