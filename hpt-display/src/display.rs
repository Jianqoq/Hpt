use hpt_common::utils::pointer::Pointer;
use hpt_traits::tensor::{CommonBounds, TensorInfo};
use hpt_types::into_scalar::Cast;
use std::fmt::Formatter;

use crate::formats::format_val;

/// # Internal Function
/// Pushes the string representation of the tensor to the string.
fn main_loop_push_str<U, T>(
    tensor: &U,
    lr_elements_size: usize,
    inner_loop: usize,
    last_stride: i64,
    string: &mut String,
    precision: usize,
    col_width: &mut Vec<usize>,
    prg: &mut Vec<i64>,
    shape: &Vec<i64>,
    mut ptr: Pointer<T>,
) where
    U: TensorInfo<T>,
    T: CommonBounds + Cast<f64>,
{
    let print = |string: &mut String, ptr: Pointer<T>, offset: &mut i64, col: usize| {
        let val = format_val(ptr[*offset], precision);
        string.push_str(&format!("{:>width$}", val, width = col_width[col]));
        if col < inner_loop - 1 {
            string.push(' ');
        }
        *offset += last_stride;
    };
    let mut outer_loop = 1;
    for i in tensor.shape().iter().take(tensor.ndim() - 1) {
        if i > &(2 * (lr_elements_size as i64)) {
            outer_loop *= 2 * (lr_elements_size as i64);
        } else {
            outer_loop *= i;
        }
    }
    for _ in 0..outer_loop {
        let mut offset = 0;
        if inner_loop >= 2 * lr_elements_size {
            for i in 0..2 {
                for j in 0..lr_elements_size {
                    print(string, ptr.clone(), &mut offset, j);
                }
                if i == 0 {
                    string.push_str("... ");
                    offset += last_stride * ((inner_loop as i64) - 2 * (lr_elements_size as i64));
                }
            }
        } else {
            for j in 0..inner_loop {
                print(string, ptr.clone(), &mut offset, j);
            }
        }
        string.push_str("]");
        for k in (0..tensor.ndim() - 1).rev() {
            if prg[k] < shape[k] {
                prg[k] += 1;
                ptr.offset(tensor.strides()[k]);
                if tensor.shape()[k] > 2 * (lr_elements_size as i64)
                    && prg[k] == (lr_elements_size as i64)
                {
                    string.push_str("\n");
                    string.push_str(&" ".repeat(k + 1 + "Tensor(".len()));
                    string.push_str("...");
                    string.push_str("\n\n");
                    string.push_str(&" ".repeat(k + 1 + "Tensor(".len()));
                    string.push_str(&"[".repeat(tensor.ndim() - (k + 1)));
                    ptr.offset(
                        tensor.strides()[k] * (tensor.shape()[k] - 2 * (lr_elements_size as i64)),
                    );
                    prg[k] += tensor.shape()[k] - 2 * (lr_elements_size as i64);
                    assert!(prg[k] < tensor.shape()[k]);
                    break;
                }

                string.push_str("\n");
                string.push_str(&" ".repeat(k + 1 + "Tensor(".len()));
                string.push_str(&"[".repeat(tensor.ndim() - (k + 1)));
                assert!(prg[k] < tensor.shape()[k]);
                break;
            } else {
                prg[k] = 0;
                string.push_str("]");
                if k >= 1 && prg[k - 1] < shape[k - 1] {
                    string.push_str(&"\n".repeat(tensor.ndim() - (k + 1)));
                }
                ptr.offset(-tensor.strides()[k] * shape[k]);
            }
        }
    }
}

/// # Internal Function
/// Get the width of each column in the tensor.
fn main_loop_get_width<U, T>(
    tensor: &U,
    lr_elements_size: usize,
    inner_loop: usize,
    last_stride: i64,
    precision: usize,
    col_width: &mut Vec<usize>,
    prg: &mut Vec<i64>,
    shape: &Vec<i64>,
    mut ptr: Pointer<T>,
) where
    U: TensorInfo<T>,
    T: CommonBounds + Cast<f64>,
{
    let mut outer_loop = 1;
    for i in tensor.shape().iter().take(tensor.ndim() - 1) {
        if i > &(2 * (lr_elements_size as i64)) {
            outer_loop *= 2 * (lr_elements_size as i64);
        } else {
            outer_loop *= i;
        }
    }
    for _ in 0..outer_loop {
        let mut offset: i64 = 0;
        if inner_loop >= 2 * lr_elements_size {
            for i in 0..2 {
                for j in 0..lr_elements_size {
                    let val = format_val(ptr[offset], precision);
                    col_width[j] = std::cmp::max(col_width[j], val.len());
                    offset += last_stride;
                }
                if i == 0 {
                    offset += last_stride * ((inner_loop as i64) - 2 * (lr_elements_size as i64));
                }
            }
        } else {
            for j in 0..inner_loop {
                let val = format_val(ptr[offset], precision);
                col_width[j] = std::cmp::max(col_width[j], val.len());
                offset += last_stride;
            }
        }
        for k in (0..tensor.ndim() - 1).rev() {
            if prg[k] < shape[k] {
                prg[k] += 1;
                ptr.offset(tensor.strides()[k]);
                if tensor.shape()[k] > 2 * (lr_elements_size as i64)
                    && prg[k] == (lr_elements_size as i64)
                {
                    ptr.offset(
                        tensor.strides()[k] * (tensor.shape()[k] - 2 * (lr_elements_size as i64)),
                    );
                    prg[k] += tensor.shape()[k] - 2 * (lr_elements_size as i64);
                    assert!(prg[k] < tensor.shape()[k]);
                    break;
                }
                assert!(prg[k] < tensor.shape()[k]);
                break;
            } else {
                prg[k] = 0;
                ptr.offset(-tensor.strides()[k] * shape[k]);
            }
        }
    }
}

/// Display a tensor.
///
/// # Arguments
/// - `tensor`: A reference to the tensor to be displayed.
/// - `f`: A reference to the formatter.
/// - `lr_elements_size`: Number of elements to display in left and right for each row and column.
/// - `precision`: Number of decimal places to display for floating point numbers.
/// - `show_backward`: A boolean indicating whether to display the gradient function of the tensor, currently only used in DiffTensor.
pub fn display<U, T>(
    tensor: U,
    f: &mut Formatter<'_>,
    lr_elements_size: usize,
    precision: usize,
    show_backward: bool,
) -> std::fmt::Result
where
    U: TensorInfo<T>,
    T: CommonBounds + Cast<f64>,
{
    let mut string: String = String::new();
    if tensor.size() == 0 {
        write!(f, "{}", "Tensor([])\n".to_string())
    } else if tensor.ndim() == 0 {
        let val = format_val(unsafe { tensor.ptr().ptr.read() }, precision);
        write!(f, "{}", format!("Tensor({})\n", val))
    } else {
        let ptr: Pointer<T> = tensor.ptr();
        if !ptr.ptr.is_null() {
            let inner_loop: usize = tensor.shape()[tensor.ndim() - 1] as usize;
            let mut prg: Vec<i64> = vec![0; tensor.ndim()];
            let mut shape: Vec<i64> = tensor.shape().to_vec();
            shape.iter_mut().for_each(|x: &mut i64| {
                *x -= 1;
            });
            let mut strides: Vec<i64> = tensor.strides().to_vec();
            shape.iter().enumerate().for_each(|(i, x)| {
                if *x == 0 {
                    strides[i] = 0;
                }
            });
            let last_stride = strides[tensor.ndim() - 1];
            string.push_str("Tensor(");
            for _ in 0..tensor.ndim() {
                string.push_str("[");
            }
            let mut col_width: Vec<usize> = vec![0; inner_loop];
            main_loop_get_width(
                &tensor,
                lr_elements_size,
                inner_loop,
                last_stride,
                precision,
                &mut col_width,
                &mut prg,
                &shape,
                ptr.clone(),
            );
            main_loop_push_str(
                &tensor,
                lr_elements_size,
                inner_loop,
                last_stride,
                &mut string,
                precision,
                &mut col_width,
                &mut prg,
                &shape,
                ptr.clone(),
            );
        }
        let shape_str = tensor
            .shape()
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join(", ");
        let strides_str = tensor
            .strides()
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join(", ");
        if !show_backward {
            string.push_str(&format!(
                ", shape=({}), strides=({}), dtype={})\n",
                shape_str,
                strides_str,
                T::STR
            ));
        } else {
            string.push_str(&format!(
                ", shape=({}), strides=({}), dtype={}, grad_fn={})\n",
                shape_str,
                strides_str,
                T::STR,
                "None"
            ));
        }
        write!(f, "{}", format!("{}", string))
    }
}
