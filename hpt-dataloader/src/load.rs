use std::{ collections::HashMap, fs::File, io::{ Read, Seek } };

use crate::{ data_loader::{ HeaderInfo, LoadResult }, CompressionAlgo };
use flate2::read::{ DeflateDecoder, GzDecoder, ZlibDecoder };
use hpt_common::slice::slice_process;

pub(crate) fn load_compressed_slice<'a>(
    file_name: &str,
    queries: Vec<(String, Vec<(i64, i64, i64)>, String, usize)>
) -> Result<HashMap<String, LoadResult>, Box<dyn std::error::Error>> {
    let res = HeaderInfo::parse_header_compressed(file_name)?;
    let mut ret = HashMap::with_capacity(res.len());
    let mut file = File::open(file_name)?;
    for (name, slices, dtype, sizeof) in queries {
        let info = res.get(&name).expect(&format!("{} not found in header", name));
        if info.dtype != dtype {
            return Err(
                Box::new(
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "the dtype stored is {}, but the dtype requested is {}",
                            info.dtype,
                            dtype
                        )
                    )
                )
            );
        }

        let chunk_size: usize = info.indices[0].3; // since the chunk size is the same for all the indices, except the last one, we can use the first one to get the chunk size

        // scale the shape, strides with mem_size
        let (res_shape, res_strides, offset) = slice_process(
            info.shape.clone(),
            info.strides.clone(),
            &slices,
            sizeof as i64
        )?;

        let mut strides = res_strides.clone();
        strides.iter_mut().for_each(|x| {
            *x /= sizeof as i64;
        });

        // since the shape is scaled with mem_size, we need to scale it back
        let mut shape = res_shape.clone();
        let mem_layout = std::alloc::Layout
            ::from_size_align(shape.iter().product::<i64>() as usize, 128)
            .unwrap();
        shape.iter_mut().for_each(|x| {
            *x /= sizeof as i64;
        });
        let ptr = unsafe { std::alloc::alloc(mem_layout) };
        let res_slice = unsafe {
            std::slice::from_raw_parts_mut(ptr as *mut u8, mem_layout.size())
        };
        if res_slice.len() == 0 {
            let res = LoadResult {
                ptr,
                shape,
                strides,
                dtype,
            };
            ret.insert(name, res);
            continue;
        }
        // the size of the inner loop
        let inner_loop_size = (*info.shape.last().unwrap() as usize) * sizeof;

        // based on the offset, get the start of the row
        let row_idx = offset / (inner_loop_size as i64);
        let num_rows_per_chunk = chunk_size / inner_loop_size;
        let mut block_idx = (row_idx as usize) / num_rows_per_chunk;

        // get buffer index, compressed data offset, compressed data size, bytes of the data in the buffer
        let (block_locate_row, idx, compressed_len, mut block_mem_size) = info.indices[block_idx];
        let mut uncompressed_data = uncompress_data(
            &mut file,
            idx as u64,
            compressed_len,
            block_mem_size,
            info.compress_algo
        )?;
        // calculate row index within the chunk
        let local_row_offset = (row_idx as usize) - block_locate_row;
        let local_col_offset = offset % (inner_loop_size as i64);

        let mut prg = vec![0; shape.len()];
        let inner = *shape.last().unwrap();
        let outer = shape.iter().product::<i64>() / inner;

        let mut start_idx = (local_row_offset * inner_loop_size +
            (local_col_offset as usize)) as i64;

        for idx in 0..(outer * inner) as usize {
            if start_idx < 0 {
                let jumped_eles2 = -start_idx;
                let jumped_rows2 = jumped_eles2 / (inner_loop_size as i64);
                let num_blocks2 = (jumped_rows2 as usize).div_ceil(num_rows_per_chunk);
                assert!(num_blocks2 <= block_idx);
                block_idx -= num_blocks2;
                start_idx =
                    ((num_blocks2 as i64) * (num_rows_per_chunk as i64) - jumped_rows2) *
                        (inner_loop_size as i64) +
                    local_col_offset;
                let (_, new_idx, new_compressed_len, new_block_mem_size) = info.indices[block_idx];
                block_mem_size = new_block_mem_size;
                uncompressed_data = uncompress_data(
                    &mut file,
                    new_idx as u64,
                    new_compressed_len,
                    new_block_mem_size,
                    info.compress_algo
                )?;
            } else if start_idx >= (block_mem_size as i64) {
                let jumped_eles = start_idx - (block_mem_size as i64);
                let jumped_rows = jumped_eles / (inner_loop_size as i64);
                let num_blocks = (jumped_rows as usize) / num_rows_per_chunk;
                block_idx += num_blocks + 1;
                start_idx =
                    (jumped_rows - (num_blocks as i64) * (num_rows_per_chunk as i64)) *
                        (inner_loop_size as i64) +
                    local_col_offset;
                let (_, new_idx, new_compressed_len, new_block_mem_size) = info.indices[block_idx];
                block_mem_size = new_block_mem_size;
                uncompressed_data = uncompress_data(
                    &mut file,
                    new_idx as u64,
                    new_compressed_len,
                    new_block_mem_size,
                    info.compress_algo
                )?;
            }
            let val = &uncompressed_data[start_idx as usize..(start_idx as usize) + sizeof];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    val.as_ptr(),
                    ptr.add(idx * sizeof),
                    sizeof
                );
            }
            for j in (0..shape.len()).rev() {
                if prg[j] < shape[j] - 1 {
                    prg[j] += 1;
                    start_idx += strides[j] * (sizeof as i64);
                    break;
                } else {
                    prg[j] = 0;
                    start_idx -= strides[j] * (shape[j] - 1) * (sizeof as i64);
                }
            }
        }
        println!("tensor load finished");
        let res = LoadResult {
            ptr,
            shape,
            strides,
            dtype,
        };
        ret.insert(name, res);
    }
    Ok(ret)
}

fn uncompress_data(
    file: &mut File,
    idx: u64,
    compressed_len: usize,
    block_mem_size: usize,
    compression_type: CompressionAlgo
) -> std::io::Result<Vec<u8>> {
    file.seek(std::io::SeekFrom::Start(idx))?;
    let mut compressed_vec = vec![0u8; compressed_len];
    file.read_exact(&mut compressed_vec)?;
    let mut uncompressed_data = vec![0u8; block_mem_size];
    match compression_type {
        CompressionAlgo::Gzip => {
            let mut decoder = GzDecoder::new(compressed_vec.as_slice());
            decoder.read_exact(&mut uncompressed_data)?;
        }
        CompressionAlgo::Deflate => {
            let mut decoder = DeflateDecoder::new(compressed_vec.as_slice());
            decoder.read_exact(&mut uncompressed_data)?;
        }
        CompressionAlgo::Zlib => {
            let mut decoder = ZlibDecoder::new(compressed_vec.as_slice());
            decoder.read_exact(&mut uncompressed_data)?;
        }
        CompressionAlgo::NoCompression => {
            uncompressed_data = compressed_vec;
        }
    }
    Ok(uncompressed_data)
}
