use crate::{compression_trait::Meta, CHUNK_BUFF};

pub fn generate_header_compressed(meta: &Meta) -> (String, (String, usize, usize, usize, usize)) {
    let infos = format!(
        r#""{}":{{"begin":FASTTENSOR_PLACEHOLD,"name":"{}","shape":{:?},"strides":{:?},"size":{},"dtype":"{}","compress_algo":{},"endian":{},"indices":place_holder0}}"#,
        meta.name,
        meta.name,
        meta.data_saver.shape().to_vec(),
        meta.data_saver.shape().to_strides().to_vec(),
        meta.data_saver.size(),
        meta.data_saver.dtype(),
        serde_json::to_string(&meta.compression_algo)
            .expect("Failed to serialize compression_algo"),
        serde_json::to_string(&meta.endian).expect("Failed to serialize endian")
    );
    let res = {
        let x = &meta.data_saver;
        let outer = x.size() / (*x.shape().last().unwrap() as usize);
        let inner = (*x.shape().last().unwrap() as usize) * x.mem_size();
        let num_chunks;
        let mut num_lines;
        let mut remain = 0;
        let mut buffer_size;
        if x.size() * x.mem_size() < CHUNK_BUFF {
            num_chunks = 1;
            num_lines = outer;
            buffer_size = num_lines * inner;
        } else {
            buffer_size = ((CHUNK_BUFF - 1) / inner) * inner;
            num_lines = buffer_size / inner;
            if num_lines == 0 {
                num_lines = 1;
                buffer_size = inner;
            }
            remain = outer % num_lines;
            num_chunks = outer / num_lines;
        }
        (
            meta.name.clone(),
            num_chunks,
            num_lines,
            remain,
            buffer_size,
        )
    };

    (infos, res)
}
