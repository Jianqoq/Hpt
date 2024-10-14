use std::io::Write;

use flate2::Compression;

pub trait CompressionTrait<const L: u32> {
    fn new() -> Self;
    fn clear_buffer(&mut self);
    fn get_inner(&mut self) -> &Vec<u8>;
    fn get_mut(&mut self) -> &mut Vec<u8>;
    fn write_all_data(&mut self, buf: &[u8]) -> std::io::Result<()>;
    fn flush_all(&mut self) -> std::io::Result<()>;
    fn id() -> &'static str;
}

macro_rules! impl_compression_trait {
    ($([$name:expr, $($t:ty),*]),*) => {
        $(
            impl<const L: u32> CompressionTrait<L> for $($t)* {
                fn new() -> Self {
                    Self::new(Vec::new(), Compression::new(L))
                }
                fn clear_buffer(&mut self) {
                    self.get_mut().clear();
                }
                fn get_inner(&mut self) -> &Vec<u8> {
                    self.get_ref()
                }
                fn get_mut(&mut self) -> &mut Vec<u8> {
                    self.get_mut()
                }
                fn write_all_data(&mut self, buf: &[u8]) -> std::io::Result<()> {
                    self.write_all(buf)
                }
                fn flush_all(&mut self) -> std::io::Result<()> {
                    self.flush()
                }
                fn id() -> &'static str {
                    $name
                }
            }
        )*
    };
}

impl_compression_trait!(
    ["0", flate2::write::GzEncoder<Vec<u8>>],
    ["1", flate2::write::DeflateEncoder<Vec<u8>>],
    ["2", flate2::write::ZlibEncoder<Vec<u8>>]
);
