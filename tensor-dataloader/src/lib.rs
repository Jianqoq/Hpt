mod compression_trait;
pub mod data_loader;
pub extern crate tensor_macros;
pub use compression_trait::{DataLoader, TensorSaver, TensorLoader};
pub use flate2::write::{DeflateEncoder, GzEncoder, ZlibEncoder};
// pub use data_loader::*;
pub use tensor_macros::*;
pub use data_loader::Endian;
pub use compression_trait::CompressionAlgo;

pub(crate) mod save;
pub(crate) mod load_compressed_slice;
pub(crate) mod utils;

pub(crate) const CHUNK_BUFF: usize = 1024 * 1024;