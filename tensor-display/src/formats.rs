use tensor_traits::CommonBounds;
use tensor_types::half::{ self, bf16 };

pub(crate) fn format_float<T: CommonBounds>(val: T, precision: usize) -> String {
    let mut val = val.to_string();
    match T::ID {
        tensor_types::dtype::Dtype::BF16 => {
            let tmp_val: bf16 = val.parse::<bf16>().expect("Failed to parse bf16");
            let f64_val = tmp_val.to_f64();
            if f64_val - (f64_val as i64 as f64) != 0.0 {
                val = format!("{:.*}", precision, val);
            } else {
                val = format!("{}.", val);
            }
        }
        tensor_types::dtype::Dtype::F16 => {
            let tmp_val: half::f16 = val.parse::<half::f16>().expect("Failed to parse f16");
            let f64_val = tmp_val.to_f64();
            if f64_val - (f64_val as i64 as f64) != 0.0 {
                val = format!("{:.*}", precision, val);
            } else {
                val = format!("{}.", val);
            }
        }
        tensor_types::dtype::Dtype::F32 => {
            let tmp_val: f32 = val.parse::<f32>().expect("Failed to parse f32");
            if tmp_val - (tmp_val as i64 as f32) != 0.0 {
                val = format!("{:.*}", precision, val);
            } else {
                val = format!("{}.", val);
            }
        }
        tensor_types::dtype::Dtype::F64 => {
            let tmp_val: f64 = val.parse::<f64>().expect("Failed to parse f64");
            if tmp_val - (tmp_val as i64 as f64) != 0.0 {
                val = format!("{:.*}", precision, val);
            } else {
                val = format!("{}.", val);
            }
        }
        _ => panic!("{} is not a floating point type", T::ID),
    }
    val
}

pub(crate) fn format_complex<T: CommonBounds>(val: T, precision: usize) -> String {
    let mut val = val.to_string();
    match T::ID {
        tensor_types::dtype::Dtype::C32 => {
            let tmp_val: num_complex::Complex32 = val
                .parse::<num_complex::Complex32>()
                .expect("Failed to parse c32");
            let re = tmp_val.re;
            let im = tmp_val.im;
            let re_str = format_float(re, precision);
            let im_str = format_float(im, precision);
            val = format!("{} + {}i", re_str, im_str);
        }
        tensor_types::dtype::Dtype::C64 => {
            let tmp_val: num_complex::Complex64 = val
                .parse::<num_complex::Complex64>()
                .expect("Failed to parse c64");
            let re = tmp_val.re;
            let im = tmp_val.im;
            let re_str = format_float(re, precision);
            let im_str = format_float(im, precision);
            val = format!("{} + {}i", re_str, im_str);
        }
        _ => panic!("{} is not a complex type", T::ID),
    }
    val
}

pub(crate) fn format_val<T: CommonBounds>(val: T, precision: usize) -> String {
    match T::ID {
        | tensor_types::dtype::Dtype::BF16
        | tensor_types::dtype::Dtype::F16
        | tensor_types::dtype::Dtype::F32
        | tensor_types::dtype::Dtype::F64 => format_float(val, precision),
        tensor_types::dtype::Dtype::C32 | tensor_types::dtype::Dtype::C64 => {
            format_complex(val, precision)
        }
        _ => val.to_string(),
    }
}

#[test]
fn test_complex() {
    let val = num_complex::Complex32::new(1.0, 2.0);
    let val = format_complex(val, 2);
    println!("{}", val);
}
