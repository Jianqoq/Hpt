pub mod reginfo;
pub use reginfo::RegisterInfo;
#[cfg(feature = "cuda")]
include!(concat!(env!("OUT_DIR"), "/generated_constants.rs"));
