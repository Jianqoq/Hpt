use std::fmt::Formatter;

use anyhow::Result;
use num_complex::Complex32;

use tensor_common::pointer::Pointer;
use tensor_traits::tensor::{CommonBounds, TensorInfo};
use tensor_types::dtype::Dtype;

const EPSILON: f64 = 1e-5;

/// # Internal Function
/// Pushes the string representation of the tensor to the string.
fn main_loop_push_str<U, T>(
    tensor: &U,
    outer_loop: usize,
    inner_loop: usize,
    last_stride: i64,
    col_exceeded: bool,
    row_exceeded: bool,
    row_threshold: usize,
    string: &mut String,
    max_element: usize,
    _: usize,
    precision: usize,
    col_width: &mut Vec<usize>,
    prg: &mut Vec<i64>,
    shape: &Vec<i64>,
    mut ptr: Pointer<T>,
)
    -> Result<()>
where
    U: TensorInfo<T>,
    T: CommonBounds,
{
    for _ in 0..outer_loop {
        let mut offset = 0;
        let mut col = 0;
        for j in 0..inner_loop {
            if col_exceeded && j + 1 == max_element / 2 {
                let mut val = ptr[offset].to_string();
                if T::ID == Dtype::F32 || T::ID == Dtype::F64 {
                    // the val is float
                    let tmp_val: f64 = val.parse::<f64>().unwrap();
                    if tmp_val - (tmp_val as i64 as f64) > EPSILON {
                        // if the float number has decimal part and the decimal part is not zero
                        // then we cut the decimal part with precision
                        val = format!("{:.*}", precision, ptr[offset]);
                    } else {
                        val = format!("{:.0}", ptr[offset]);
                    }
                } else if T::ID == Dtype::C32 || T::ID == Dtype::C64 {
                    let tmp_val: Complex32 = val.parse::<Complex32>().unwrap();
                    if
                    tmp_val.re - (tmp_val.re as i64 as f32) != 0.0 ||
                        tmp_val.im - (tmp_val.im as i64 as f32) != 0.0
                    {
                        val = format!("{:.*}", precision, ptr[offset]);
                    }
                }
                string.push_str(&format!("{:>width$} ... ", val, width = col_width[j]));
                offset +=
                    (((tensor.shape()[tensor.ndim() - 1] as usize) - max_element + 1) as i64) *
                        last_stride;
            } else {
                let mut val = ptr[offset].to_string();
                if T::ID == Dtype::F32 || T::ID == Dtype::F64 {
                    // the val is float
                    let tmp_val: f64 = val.parse::<f64>()?;
                    if tmp_val - (tmp_val as i64 as f64) != 0.0 {
                        // if the float number has decimal part and the decimal part is not zero
                        // then we cut the decimal part with precision
                        val = format!("{:.*}", precision, ptr[offset]);
                    } else {
                        val = format!("{:.0}", ptr[offset]);
                    }
                } else if T::ID == Dtype::C32 || T::ID == Dtype::C64 {
                    let tmp_val: Complex32 = val.parse::<Complex32>().unwrap();
                    if
                    tmp_val.re - (tmp_val.re as i64 as f32) != 0.0 ||
                        tmp_val.im - (tmp_val.im as i64 as f32) != 0.0
                    {
                        val = format!("{:.*}", precision, ptr[offset]);
                    }
                }
                if j + 1 == inner_loop {
                    string.push_str(&format!("{:>width$}", val, width = col_width[col]));
                } else {
                    // if j > max_elemnt_each_row && inner_loop - j > max_elemnt_each_row / 2 {
                    //     string.push_str("\n");
                    // }
                    string.push_str(&format!("{:>width$} ", val, width = col_width[col]));
                }
                offset += last_stride;
            }
            if !col_exceeded && (j + 1) % row_threshold == 0 {
                string.push_str("\n");
                string.push_str(&" ".repeat("Tensor(".len() + (tensor.ndim())));
                col = 0;
            } else {
                col += 1;
            }
        }
        string.push_str("]");
        for k in (0..(tensor.ndim()) - 1).rev() {
            if prg[k] < shape[k] {
                if
                row_exceeded &&
                    prg[k] + 1 == (max_element as i64) / 2 &&
                    tensor.shape()[k] > (max_element as i64)
                {
                    prg[k] += tensor.shape()[k] - (max_element as i64) + 1;
                    string.push_str("\n");
                    string.push_str(&" ".repeat("Tensor(".len() + (tensor.ndim())));
                    string.push_str("... \n");
                    string.push_str(&" ".repeat(k + 1 + "Tensor(".len()));
                    string.push_str(&"[".repeat((tensor.ndim()) - (k + 1)));
                    ptr.offset(
                        tensor.strides()[k] *
                            tensor.shape()[k] - (max_element as i64) + 1
                    );
                    assert!(prg[k] < tensor.shape()[k]);
                } else {
                    prg[k] += 1;
                    string.push_str("\n");
                    string.push_str(&" ".repeat(k + 1 + "Tensor(".len()));
                    string.push_str(&"[".repeat((tensor.ndim()) - (k + 1)));
                    ptr.offset(tensor.strides()[k]);
                    assert!(prg[k] < tensor.shape()[k]);
                }
                break;
            } else {
                prg[k] = 0;
                string.push_str("]");
                if k >= 1 && prg[k - 1] < shape[k - 1] {
                    string.push_str(&"\n".repeat((tensor.ndim()) - (k + 1)));
                }
                ptr.offset(-tensor.strides()[k] * (shape[k]));
            }
        }
    }
    Ok(())
}

/// # Internal Function
/// Get the width of each column in the tensor.
fn main_loop_get_width<U, T>(
    tensor: &U,
    outer_loop: usize,
    inner_loop: usize,
    last_stride: i64,
    col_exceeded: bool,
    row_exceeded: bool,
    max_element: usize,
    row_threshold: usize,
    _: usize,
    precision: usize,
    col_width: &mut Vec<usize>,
    prg: &mut Vec<i64>,
    shape: &Vec<i64>,
    mut ptr: Pointer<T>,
)
    -> Result<()>
where
    U: TensorInfo<T>,
    T: CommonBounds,
{
    for _ in 0..outer_loop {
        let mut offset: i64 = 0;
        let mut col = 0;
        for j in 0..inner_loop {
            if col_exceeded && j + 1 == max_element / 2 {
                let mut val: String = ptr[offset].to_string();
                if T::ID == Dtype::F32 || T::ID == Dtype::F64 {
                    let tmp_val: f64 = val.parse::<f64>().unwrap();
                    if tmp_val - (tmp_val as i64 as f64) < EPSILON {
                        val = format!("{:.0}", ptr[offset]);
                    } else {
                        val = format!("{:.*}", precision, ptr[offset]);
                    }
                    col_width[j] = std::cmp::max(col_width[j], val.len());
                } else if T::ID == Dtype::C32 || T::ID == Dtype::C64 {
                    let tmp_val: Complex32 = val.parse::<Complex32>().unwrap();
                    if
                    tmp_val.re - (tmp_val.re as i64 as f32) != 0.0 ||
                        tmp_val.im - (tmp_val.im as i64 as f32) != 0.0
                    {
                        val = format!("{:.*}", precision, ptr[offset]);
                    }
                    col_width[j] = std::cmp::max(col_width[j], val.len());
                } else {
                    col_width[j] = std::cmp::max(col_width[j], val.len());
                }
                offset +=
                    tensor.shape()[(tensor.ndim()) - 1] -
                            (max_element as i64) +
                            1 * last_stride;
            } else {
                let mut val: String = ptr[offset].to_string();
                if T::ID == Dtype::F32 || T::ID == Dtype::F64 {
                    // the val is float
                    let tmp_val: f64 = val.parse::<f64>()?;
                    if tmp_val - (tmp_val as i64 as f64) < EPSILON {
                        val = format!("{:.0}", ptr[offset]);
                    } else {
                        val = format!("{:.*}", precision, ptr[offset]);
                    }
                    col_width[col] = std::cmp::max(col_width[col], val.len());
                } else if T::ID == Dtype::C32 || T::ID == Dtype::C64 {
                    let tmp_val: Complex32 = val.parse::<Complex32>().unwrap();
                    if
                    tmp_val.re - (tmp_val.re as i64 as f32) != 0.0 ||
                        tmp_val.im - (tmp_val.im as i64 as f32) != 0.0
                    {
                        val = format!("{:.*}", precision, ptr[offset]);
                    }
                    col_width[col] = std::cmp::max(col_width[col], val.len());
                } else {
                    col_width[col] = std::cmp::max(col_width[col], val.len());
                }
                offset += last_stride;
            }
            if !col_exceeded && (col + 1) % row_threshold == 0 {
                col = 0;
            } else {
                col += 1;
            }
        }
        for k in (0..(tensor.ndim()) - 1).rev() {
            if prg[k] < shape[k] {
                if
                row_exceeded &&
                    prg[k] + 1 == (max_element as i64) / 2 &&
                    tensor.shape()[k] > (max_element as i64)
                {
                    prg[k] += tensor.shape()[k] - (max_element as i64) + 1;
                    ptr.offset(
                        tensor.strides()[k] *
                            tensor.shape()[k] - (max_element as i64) + 1
                    );
                    assert!(prg[k] < tensor.shape()[k]);
                } else {
                    prg[k] += 1;
                    ptr.offset(tensor.strides()[k]);
                    assert!(prg[k] < tensor.shape()[k]);
                }
                break;
            } else {
                prg[k] = 0;
                ptr.offset(-tensor.strides()[k] * (shape[k]));
            }
        }
    }
    Ok(())
}

/// Display a tensor.
///
/// # Arguments
/// - `tensor`: A reference to the tensor to be displayed.
/// - `f`: A reference to the formatter.
/// - `max_size`: Threshold for displaying partial data of the tensor.
/// - `max_element`: Number of elements to display in each row when the size of the tensor exceeds `max_size`.
/// - `max_elemnt_each_row`: Number of elements to display in each row when the size of the tensor exceeds `max_size`.
/// - `precision`: Number of decimal places to display for floating point numbers.
/// - `show_backward`: A boolean indicating whether to display the gradient function of the tensor, currently only used in DiffTensor.
pub fn display<U, T>(
    tensor: U,
    f: &mut Formatter<'_>,
    max_size: usize,
    max_element: i64,
    max_elemnt_each_row: i64,
    row_threshold: usize,
    precision: usize,
    show_backward: bool,
)
    -> std::fmt::Result
where
    U: TensorInfo<T>,
    T: CommonBounds,
{
    let mut string: String = String::new();
    return if tensor.size() == 0 {
        write!(f, "{}", "Tensor([])\n".to_string())
    } else if tensor.ndim() == 0 {
        write!(f, "{}", format!("Tensor({})\n", tensor.ptr().read()))
    } else {
        let ptr: Pointer<T> = tensor.ptr();
        if !ptr.ptr.is_null() {
            let mut inner_loop: usize = tensor.shape()[(tensor.ndim()) - 1] as usize;
            let mut outer_loop: usize = (tensor.size()) / inner_loop;
            let mut prg: Vec<i64> = vec![0; tensor.ndim()];
            let mut shape: Vec<i64> = tensor.shape().to_vec();
            shape.iter_mut().for_each(|x: &mut i64| {
                *x -= 1;
            });
            let mut strides: Vec<i64> = tensor.strides().to_vec();
            shape
                .iter()
                .enumerate()
                .for_each(|(i, x)| {
                    if *x == 0 {
                        strides[i] = 0;
                    }
                });
            let last_stride = strides[(tensor.ndim()) - 1];
            let mut col_exceeded: bool = false;
            let mut row_exceeded: bool = false;
            if tensor.size() > max_size {
                if tensor.shape()[(tensor.ndim()) - 1] >= max_element {
                    inner_loop = max_element as usize;
                    col_exceeded = true;
                }
                let tmp: &[i64] = &tensor.shape()[..(tensor.ndim()) - 1];
                let acc = tmp.iter().fold(1, |acc, x| {
                    if *x > max_element { max_element * acc } else { acc * x }
                });
                outer_loop = acc as usize;
                if outer_loop >= (max_element as usize) {
                    row_exceeded = true;
                }
            }
            string.push_str("Tensor(");
            for _ in 0..tensor.ndim() {
                string.push_str("[");
            }
            let mut col_width: Vec<usize> = vec![0; inner_loop];
            main_loop_get_width(
                &tensor,
                outer_loop,
                inner_loop,
                last_stride,
                col_exceeded,
                row_exceeded,
                max_element as usize,
                row_threshold,
                max_elemnt_each_row as usize,
                precision,
                &mut col_width,
                &mut prg,
                &shape,
                ptr.clone(),
            ).unwrap();
            main_loop_push_str(
                &tensor,
                outer_loop,
                inner_loop,
                last_stride,
                col_exceeded,
                row_exceeded,
                row_threshold,
                &mut string,
                max_element as usize,
                max_elemnt_each_row as usize,
                precision,
                &mut col_width,
                &mut prg,
                &shape,
                ptr.clone(),
            ).unwrap();
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
            string.push_str(
                &format!(", shape=({}), strides=({}), dtype={})\n", shape_str, strides_str, T::ID)
            );
        } else {
            string.push_str(
                &format!(
                    ", shape=({}), strides=({}), dtype={}, grad_fn={})\n",
                    shape_str,
                    strides_str,
                    T::ID,
                    "None"
                )
            );
        }
        write!(f, "{}", format!("{}\n", string))
    };
}
