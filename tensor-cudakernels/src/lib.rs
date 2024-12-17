pub mod reginfo;
pub use reginfo::RegisterInfo;
include!(concat!(env!("OUT_DIR"), "/generated_constants.rs"));