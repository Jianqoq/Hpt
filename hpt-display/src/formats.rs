use hpt_traits::tensor::CommonBounds;
use hpt_types::into_scalar::Cast;

pub(crate) fn format_float<T: CommonBounds + Cast<f64>>(val: T, precision: usize) -> String {
    match T::STR {
        "bf16" => {
            let f64_val: f64 = val.cast();
            if f64_val - (f64_val as i64 as f64) != 0.0 {
                format!("{:.prec$}", f64_val, prec = precision)
            } else {
                format!("{:.0}.", f64_val)
            }
        }
        "f16" => {
            let f64_val: f64 = val.cast();
            if f64_val - (f64_val as i64 as f64) != 0.0 {
                format!("{:.prec$}", val, prec = precision)
            } else {
                format!("{:.0}.", val)
            }
        }
        "f32" => {
            let tmp_val: f64 = val.cast();
            if tmp_val - (tmp_val as i64 as f64) != 0.0 {
                format!("{:.prec$}", val, prec = precision)
            } else {
                format!("{:.0}.", val)
            }
        }
        "f64" => {
            let tmp_val: f64 = val.cast();
            if tmp_val - (tmp_val as i64 as f64) != 0.0 {
                format!("{:.prec$}", val, prec = precision)
            } else {
                format!("{:.0}.", val)
            }
        }
        _ => panic!("{} is not a floating point type", T::STR),
    }
}

pub(crate) fn format_complex<T: CommonBounds>(val: T, precision: usize) -> String {
    let mut val = val.to_string();
    match T::STR {
        "c32" => {
            let tmp_val: num_complex::Complex32 = val
                .parse::<num_complex::Complex32>()
                .expect("Failed to parse c32");
            let re = tmp_val.re;
            let im = tmp_val.im;
            let re_str = format_float(re, precision);
            let im_str = format_float(im, precision);
            val = format!("{} + {}i", re_str, im_str);
        }
        "c64" => {
            let tmp_val: num_complex::Complex64 = val
                .parse::<num_complex::Complex64>()
                .expect("Failed to parse c64");
            let re = tmp_val.re;
            let im = tmp_val.im;
            let re_str = format_float(re, precision);
            let im_str = format_float(im, precision);
            val = format!("{} + {}i", re_str, im_str);
        }
        _ => panic!("{} is not a complex type", T::STR),
    }
    val
}

pub(crate) fn format_val<T: CommonBounds + Cast<f64>>(val: T, precision: usize) -> String {
    match T::STR {
        "bf16" | "f16" | "f32" | "f64" => format_float(val, precision),

        "c32" | "c64" => format_complex(val, precision),
        _ => val.to_string(),
    }
}

#[test]
fn test_complex() {
    let val = num_complex::Complex32::new(1.0, 2.0);
    let val = format_complex(val, 2);
    println!("{}", val);
}
