
use tch::Tensor as TchTensor;
use tensor_dyn::Tensor;
use tensor_dyn::TensorInfo;
use tensor_dyn::TensorLike;

#[allow(unused)]
pub(crate) fn assert_eq(a: &TchTensor, b: &Tensor<f64>) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const f64, b.size()) };
    let b_raw = b.as_raw();
    let tolerance = 2.5e-15;

    for i in 0..b.size() {
        let abs_diff = (a_raw[i] - b_raw[i]).abs();
        let relative_diff = abs_diff / b_raw[i].abs().max(f64::EPSILON);
        
        if abs_diff > tolerance && relative_diff > tolerance {
            panic!("{} != {} (abs_diff: {}, relative_diff: {})", a_raw[i], b_raw[i], abs_diff, relative_diff);
        }
    }
}

#[allow(unused)]
pub(crate) fn assert_eq_print(a: &TchTensor, b: &Tensor<f64>) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const f64, b.size()) };
    let b_raw = b.as_raw();
    let tolerance = 2.5e-15;

    for i in 0..b.size() {
        let abs_diff = (a_raw[i] - b_raw[i]).abs();
        let relative_diff = abs_diff / b_raw[i].abs().max(f64::EPSILON);
        
        if abs_diff > tolerance && relative_diff > tolerance {
            println!("{} != {} (abs_diff: {}, relative_diff: {})", a_raw[i], b_raw[i], abs_diff, relative_diff);
        }
    }
}

#[allow(unused)]
pub(crate) fn assert_eq_with_prec_print(a: &TchTensor, b: &Tensor<f64>) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const f64, b.size()) };
    let b_raw = b.as_raw();
    for i in 0..b.size() {
        if a_raw[i] < b_raw[i] - 1e-10 || a_raw[i] > b_raw[i] + 1e-10 {
            println!("{} != {}", a_raw[i], b_raw[i]);
        }
    }
    let abs_error = a_raw.iter().zip(b_raw.iter()).map(|(a, b)| (a - b).abs()).collect::<Vec<f64>>();
    let relative_error = a_raw.iter().zip(b_raw.iter()).map(|(a, b)| ((a - b).abs() / b.abs()).max(f64::EPSILON)).collect::<Vec<f64>>();
    let max_abs_error = abs_error.iter().copied().fold(f64::NAN, f64::max);
    let max_relative_error = relative_error.iter().copied().fold(f64::NAN, f64::max);
    println!("Max Abs Error: {}", max_abs_error);
    println!("Max Relative Error: {}", max_relative_error);
}