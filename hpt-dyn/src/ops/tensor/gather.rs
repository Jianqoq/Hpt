use hpt_common::error::{base::TensorError, shape::ShapeError};
use hpt_traits::tensor::TensorInfo;

use crate::Tensor;

use crate::DType;

impl Tensor {
    pub fn gather(&self, indices: &Tensor, axis: i64) -> Result<Tensor, TensorError> {
        let axis = if axis < 0 {
            (self.ndim() as i64 + axis)
                .try_into()
                .expect("axis is still negative after adding ndim")
        } else {
            axis as usize
        };

        ShapeError::check_index_out_of_range(axis, self.ndim())?;
        ShapeError::check_contiguous("Indices must be contiguous".to_string(), &indices.layout)?;

        if !matches!(indices.dtype, DType::I32 | DType::I64) {
            panic!("Indices must be integer type, got {:?}", indices.dtype);
        }
        let mut output_shape = Vec::with_capacity(self.ndim() + indices.ndim() - 1);
        for i in 0..axis {
            output_shape.push(self.shape()[i]);
        }
        output_shape.extend_from_slice(indices.shape());
        for i in (axis + 1)..self.ndim() {
            output_shape.push(self.shape()[i]);
        }
        let result = Tensor::empty(&output_shape, self.dtype, self.device.clone())?;

        match indices.dtype {
            DType::I32 => todo!(),
            DType::I64 => {
                let indices_flat = unsafe {
                    std::slice::from_raw_parts(
                        indices.data.cast::<i64>().ptr,
                        indices.layout.size() as usize,
                    )
                };

                indices_flat.into_iter().enumerate().for_each(|(i, idx)| {
                    let mut inp_selections = vec![(0, 0x7FFFFFFFFFFFFFFF, 1); self.ndim()];
                    inp_selections[axis] = (*idx, *idx + 1, 1);
                    let inp_slice = self.slice(&inp_selections).expect("slice failed");

                    let mut idx_pos = vec![0; indices.ndim()];
                    let mut flat_idx = i as i64;
                    for d in (0..indices.ndim()).rev() {
                        idx_pos[d] = flat_idx % indices.shape()[d];
                        flat_idx /= indices.shape()[d];
                    }

                    let mut result_selections = vec![(0, 0x7FFFFFFFFFFFFFFF, 1); result.ndim()];

                    for (d, &pos) in idx_pos.iter().enumerate() {
                        result_selections[axis + d] = (pos, pos + 1, 1);
                    }

                    let mut result_slice = result
                        .slice(&result_selections)
                        .expect("result slice failed");

                    result_slice._copy_from(&inp_slice, 1);
                });
            }
            _ => unreachable!(),
        }
        Ok(result)
    }
}
