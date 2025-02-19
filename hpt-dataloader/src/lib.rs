mod compression_trait;
pub mod data_loader;
pub use compression_trait::{DataLoader, TensorLoader, TensorSaver};
pub use flate2::write::{DeflateEncoder, GzEncoder, ZlibEncoder};
pub use compression_trait::CompressionAlgo;
pub use compression_trait::DataLoaderTrait;
pub use compression_trait::Meta;
pub use data_loader::Endian;
pub use from_safetensors::from_safetensors::FromSafeTensors;
pub use struct_save::gen_header;
pub use struct_save::load::{Load, MetaLoad};
pub use struct_save::save::save;
pub use struct_save::save::Save;
pub use utils::CPUTensorCreator;
mod struct_save {
    pub mod gen_header;
    pub mod load;
    pub mod save;
}

pub mod from_safetensors {
    pub mod from_safetensors;
}

pub mod load;
pub mod save;
pub mod utils;

pub(crate) const CHUNK_BUFF: usize = 1024 * 1024;
