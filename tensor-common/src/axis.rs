use std::ops::Index;

use crate::err_handler::ErrHandler;

pub trait Axis where Self: Iterator<Item = i64> + ExactSizeIterator + Index<usize, Output = i64> {
    fn process_axes(&self, ndim: usize) -> anyhow::Result<Vec<usize>> {
        let ndim = ndim as i64;
        let mut new_axes = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            if self[i] < 0 {
                let val = (self[i] % ndim) + ndim;
                new_axes.push(val as usize);
            } else {
                if self[i] >= ndim {
                    return Err(
                        ErrHandler::IndexOutOfRange(
                            format!(
                                "Axes {} out of range(Should be {}..{}). Pos: {}",
                                self[i],
                                0,
                                ndim,
                                i
                            )
                        ).into()
                    );
                }
                new_axes.push(self[i] as usize);
            }
        }
        Ok(new_axes)
    }
}
