use anyhow::Ok;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek},
};

use crate::CompressionAlgo;

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum Endian {
    Little,
    Big,
    Native,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct HeaderInfos {
    pub(crate) begin: u64,
    pub(crate) infos: Vec<HeaderInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct HeaderInfo {
    pub(crate) begin: u64,
    pub(crate) name: String,
    pub(crate) shape: Vec<i64>,
    pub(crate) strides: Vec<i64>,
    pub(crate) size: usize,
    pub(crate) indices: Vec<(usize, usize, usize, usize)>,
    pub(crate) compress_algo: CompressionAlgo,
    pub(crate) dtype: String,
    pub(crate) endian: Endian,
}
impl HeaderInfo {
    pub(crate) fn parse_header_compressed(file: &str) -> Result<HashMap<String, HeaderInfo>> {
        let mut file = File::open(file)?;
        file.read_exact(&mut [0u8; "FASTTENSOR".len()])?;
        let mut header_infos = [0u8; 20];
        file.read_exact(&mut header_infos)?;
        let header = std::str::from_utf8(&header_infos)?;
        let header_int = header.trim().parse::<u64>()?;
        file.seek(std::io::SeekFrom::Start(header_int))?; // offset for header
        let mut buffer3 = vec![];
        file.read_to_end(&mut buffer3)?;
        let info = std::str::from_utf8(&buffer3)?;
        let ret: HashMap<String, HeaderInfo> =
            serde_json::from_str::<HashMap<String, Self>>(&info)?;
        Ok(ret)
    }
}
