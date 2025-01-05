mod compression_trait;
pub mod data_loader;
pub extern crate tensor_macros;
pub use compression_trait::{DataLoader, TensorSaver, TensorLoader};
pub use flate2::write::{DeflateEncoder, GzEncoder, ZlibEncoder};
// pub use data_loader::*;
pub use tensor_macros::*;
pub use data_loader::Endian;
pub use compression_trait::CompressionAlgo;
pub use struct_save::gen_header;
pub use compression_trait::Meta;
pub use compression_trait::DataLoaderTrait;
pub use struct_save::save::Save;
pub use struct_save::load::{Load, MetaLoad};
pub use struct_save::save::save;
pub use from_safetensors::from_safetensors::FromSafeTensors;
mod struct_save {
    pub mod gen_header;
    pub mod save;
    pub mod load;
}

pub mod from_safetensors {
    pub mod from_safetensors;
}

pub mod save;
pub mod load;
pub mod utils;

pub(crate) const CHUNK_BUFF: usize = 1024 * 1024;