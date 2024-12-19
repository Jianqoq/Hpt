// use num_complex::{Complex32, Complex64};

// use crate::type_promote::NormalOut;

// use super::{convertion::CudaConvertor, scalar::Scalar};

// impl NormalOut<Scalar<bool>> for Scalar<bool> {
//     type Output = Scalar<bool>;
//     #[inline(always)]
//     fn _pow(self, rhs: Scalar<bool>) -> Self::Output {
//         self || rhs
//     }
//     #[inline(always)]
//     fn _clip(self, min: Scalar<bool>, max: Scalar<bool>) -> Self::Output {
//         Scalar::new(format!(
//             "(({} > {}) ? {} : ({} < {}) ? {} : ({} > {}) ? {} : {})",
//             min.val,
//             max.val,
//             min.val,
//             self.val,
//             min.val,
//             min.val,
//             self.val,
//             max.val,
//             max.val,
//             self.val
//         ))
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Scalar<bool>, b: Scalar<bool>) -> Self::Output {
//         Scalar::new(format!(
//             "(bool)((int){} * (int){} + (int){})",
//             self.val, a.val, b.val
//         ))
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Scalar<bool>) -> Self::Output {
//         Scalar::new(format!(
//             "(bool)((unsigned char){} + (unsigned char){})",
//             self.val, rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Scalar<bool>) -> Self::Output {
//         Scalar::new(format!("(bool)((char){} - (char){})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Scalar<bool>) -> Self::Output {
//         Scalar::new(format!(
//             "(bool)((unsigned char){} * (unsigned char){})",
//             self.val, rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Scalar<bool>) -> Self::Output {
//         Scalar::new(format!(
//             "({}) ? (bool)((int){} % 1) : false",
//             rhs.val, self.val
//         ))
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Scalar<bool>) -> Self::Output {
//         Scalar::new(format!("{} || {}", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Scalar<bool>) -> Self::Output {
//         Scalar::new(format!("{} && {}", self.val, rhs.val))
//     }
// }
// impl NormalOut<Scalar<half::f16>> for Scalar<bool> {
//     type Output = Scalar<half::f16>;
//     #[inline(always)]
//     fn _pow(self, rhs: Scalar<half::f16>) -> Self::Output {
//         self.to_f16().powf(rhs.to_f16())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Scalar<half::f16>, max: Scalar<half::f16>) -> Self::Output {
//         Scalar::new(format!(
//             "__float2half(fminf(fmaxf(__half2float({}), __half2float({})), __half2float({})))",
//             self.to_f16().val,
//             min.val,
//             max.val
//         ))
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: half::f16, b: half::f16) -> Self::Output {
//         Scalar::new(format!(
//             "__hfma({}, {}, {})",
//             self.to_f16().val,
//             a.val,
//             b.val
//         ))
//     }
//     #[inline(always)]
//     fn _add(self, rhs: half::f16) -> Self::Output {
//         self.add_ret_f32(rhs).to_f16()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: half::f16) -> Self::Output {
//         self.sub_ret_f32(rhs).to_f16()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: half::f16) -> Self::Output {
//         self.mul_ret_f32(rhs).to_f16()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: half::f16) -> Self::Output {
//         Scalar::new(format!(
//             "__float2half(fmodf(__half2float({}), __half2float({})))",
//             self.to_f16().val,
//             rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _max(self, rhs: half::f16) -> Self::Output {
//         Scalar::new(format!("__hmax({}, {})", self.to_f16().val, rhs.val))
//     }
//     #[inline(always)]
//     fn _min(self, rhs: half::f16) -> Self::Output {
//         Scalar::new(format!("__hmin({}, {})", self.to_f16().val, rhs.val))
//     }
// }
// impl NormalOut<Scalar<f32>> for Scalar<bool> {
//     type Output = Scalar<f32>;
//     #[inline(always)]
//     fn _pow(self, rhs: Scalar<f32>) -> Self::Output {
//         Scalar::new(format!("powf((float){}, {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _clip(self, min: Scalar<f32>, max: Scalar<f32>) -> Self::Output {
//         Scalar::new(format!(
//             "fminf(fmaxf((float){}, {}), {})",
//             self.val, min.val, max.val
//         ))
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Scalar<f32>, b: Scalar<f32>) -> Self::Output {
//         Scalar::new(format!("fmaf((float){}, {}, {})", self.val, a.val, b.val))
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Scalar<f32>) -> Self::Output {
//         Scalar::new(format!("((float){} + {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Scalar<f32>) -> Self::Output {
//         Scalar::new(format!("((float){} - {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Scalar<f32>) -> Self::Output {
//         Scalar::new(format!("((float){} * {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Scalar<f32>) -> Self::Output {
//         Scalar::new(format!("fmodf((float){}, {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Scalar<f32>) -> Self::Output {
//         Scalar::new(format!("fmaxf((float){}, {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Scalar<f32>) -> Self::Output {
//         Scalar::new(format!("fminf((float){}, {})", self.val, rhs.val))
//     }
// }
// impl NormalOut<Scalar<f64>> for Scalar<bool> {
//     type Output = Scalar<f64>;
//     #[inline(always)]
//     fn _pow(self, rhs: Scalar<f64>) -> Self::Output {
//         Scalar::new(format!("pow((double){}, {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _clip(self, min: Scalar<f64>, max: Scalar<f64>) -> Self::Output {
//         Scalar::new(format!(
//             "fmin(fmax((double){}, {}), {})",
//             self.val, min.val, max.val
//         ))
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Scalar<f64>, b: Scalar<f64>) -> Self::Output {
//         Scalar::new(format!("fma((double){}, {}, {})", self.val, a.val, b.val))
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Scalar<f64>) -> Self::Output {
//         Scalar::new(format!("((double){} + {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Scalar<f64>) -> Self::Output {
//         Scalar::new(format!("((double){} - {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Scalar<f64>) -> Self::Output {
//         Scalar::new(format!("((double){} * {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Scalar<f64>) -> Self::Output {
//         Scalar::new(format!("fmod((double){}, {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Scalar<f64>) -> Self::Output {
//         Scalar::new(format!("fmax((double){}, {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Scalar<f64>) -> Self::Output {
//         Scalar::new(format!("fmin((double){}, {})", self.val, rhs.val))
//     }
// }
// impl NormalOut<Scalar<i8>> for Scalar<bool> {
//     type Output = Scalar<i8>;
//     #[inline(always)]
//     fn _pow(self, rhs: Scalar<i8>) -> Self::Output {
//         Scalar::new(format!(
//             "((char)powf((float)(char){}, (float){}))",
//             self.val, rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _clip(self, min: Scalar<i8>, max: Scalar<i8>) -> Self::Output {
//         Scalar::new(format!(
//             "min(max((char){}, {}), {})",
//             self.val, min.val, max.val
//         ))
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Scalar<i8>, b: Scalar<i8>) -> Self::Output {
//         Scalar::new(format!(
//             "((char)((char){} * {} + {}))",
//             self.val, a.val, b.val
//         ))
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Scalar<i8>) -> Self::Output {
//         Scalar::new(format!("((char)((char){} + {}))", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Scalar<i8>) -> Self::Output {
//         Scalar::new(format!("((char)((char){} - {}))", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Scalar<i8>) -> Self::Output {
//         Scalar::new(format!("((char)((char){} * {}))", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Scalar<i8>) -> Self::Output {
//         Scalar::new(format!(
//             "({} != 0) ? (char)((char){} % {}) : 0",
//             rhs.val, self.val, rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Scalar<i8>) -> Self::Output {
//         Scalar::new(format!("max((char){}, {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Scalar<i8>) -> Self::Output {
//         Scalar::new(format!("min((char){}, {})", self.val, rhs.val))
//     }
// }
// impl NormalOut<Scalar<i16>> for Scalar<bool> {
//     type Output = Scalar<i16>;
//     #[inline(always)]
//     fn _pow(self, rhs: Scalar<i16>) -> Self::Output {
//         Scalar::new(format!(
//             "((short)powf((float)(short){}, (float){}))",
//             self.val, rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _clip(self, min: Scalar<i16>, max: Scalar<i16>) -> Self::Output {
//         Scalar::new(format!(
//             "min(max((short){}, {}), {})",
//             self.val, min.val, max.val
//         ))
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Scalar<i16>, b: Scalar<i16>) -> Self::Output {
//         // bool * a + b
//         Scalar::new(format!(
//             "((short)((short){} * {} + {}))",
//             self.val, a.val, b.val
//         ))
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Scalar<i16>) -> Self::Output {
//         Scalar::new(format!("((short)((short){} + {}))", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Scalar<i16>) -> Self::Output {
//         Scalar::new(format!("((short)((short){} - {}))", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Scalar<i16>) -> Self::Output {
//         Scalar::new(format!("((short)((short){} * {}))", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Scalar<i16>) -> Self::Output {
//         Scalar::new(format!(
//             "(({} != 0) ? (short)((short){} % {}) : 0)",
//             rhs.val, self.val, rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Scalar<i16>) -> Self::Output {
//         Scalar::new(format!("max((short){}, {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Scalar<i16>) -> Self::Output {
//         Scalar::new(format!("min((short){}, {})", self.val, rhs.val))
//     }
// }
// impl NormalOut<Scalar<i32>> for Scalar<bool> {
//     type Output = Scalar<i32>;
//     #[inline(always)]
//     fn _pow(self, rhs: Scalar<i32>) -> Self::Output {
//         Scalar::new(format!(
//             "((int)powf((float)(int){}, (float){}))",
//             self.val, rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _clip(self, min: Scalar<i32>, max: Scalar<i32>) -> Self::Output {
//         Scalar::new(format!(
//             "min(max((int){}, {}), {})",
//             self.val, min.val, max.val
//         ))
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Scalar<i32>, b: Scalar<i32>) -> Self::Output {
//         Scalar::new(format!("((int){} * {} + {})", self.val, a.val, b.val))
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Scalar<i32>) -> Self::Output {
//         Scalar::new(format!("((int){} + {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Scalar<i32>) -> Self::Output {
//         Scalar::new(format!("((int){} - {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Scalar<i32>) -> Self::Output {
//         Scalar::new(format!("((int){} * {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Scalar<i32>) -> Self::Output {
//         Scalar::new(format!(
//             "(({} != 0) ? ((int){} % {}) : 0)",
//             rhs.val, self.val, rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Scalar<i32>) -> Self::Output {
//         Scalar::new(format!("max((int){}, {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Scalar<i32>) -> Self::Output {
//         Scalar::new(format!("min((int){}, {})", self.val, rhs.val))
//     }
// }
// impl NormalOut<Scalar<i64>> for Scalar<bool> {
//     type Output = Scalar<i64>;
//     #[inline(always)]
//     fn _pow(self, rhs: Scalar<i64>) -> Self::Output {
//         Scalar::new(format!(
//             "((long long)pow((double)(long long){}, (double){}))",
//             self.val, rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _clip(self, min: Scalar<i64>, max: Scalar<i64>) -> Self::Output {
//         Scalar::new(format!(
//             "min(max((long long){}, {}), {})",
//             self.val, min.val, max.val
//         ))
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Scalar<i64>, b: Scalar<i64>) -> Self::Output {
//         Scalar::new(format!("((long long){} * {} + {})", self.val, a.val, b.val))
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Scalar<i64>) -> Self::Output {
//         Scalar::new(format!("((long long){} + {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Scalar<i64>) -> Self::Output {
//         Scalar::new(format!("((long long){} - {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Scalar<i64>) -> Self::Output {
//         Scalar::new(format!("((long long){} * {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Scalar<i64>) -> Self::Output {
//         // 需要处理除数为0的情况
//         Scalar::new(format!(
//             "(({} != 0) ? ((long long){} % {}) : 0)",
//             rhs.val, self.val, rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Scalar<i64>) -> Self::Output {
//         Scalar::new(format!("max((long long){}, {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Scalar<i64>) -> Self::Output {
//         Scalar::new(format!("min((long long){}, {})", self.val, rhs.val))
//     }
// }
// impl NormalOut<Scalar<u8>> for Scalar<bool> {
//     type Output = Scalar<u8>;
//     #[inline(always)]
//     fn _pow(self, rhs: Scalar<u8>) -> Self::Output {
//         Scalar::new(format!(
//             "((unsigned char)powf((float)(unsigned char){}, (float){}))",
//             self.val, rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _clip(self, min: Scalar<u8>, max: Scalar<u8>) -> Self::Output {
//         Scalar::new(format!(
//             "min(max((unsigned char){}, {}), {})",
//             self.val, min.val, max.val
//         ))
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Scalar<u8>, b: Scalar<u8>) -> Self::Output {
//         // bool * a + b
//         Scalar::new(format!(
//             "((unsigned char)((unsigned char){} * {} + {}))",
//             self.val, a.val, b.val
//         ))
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Scalar<u8>) -> Self::Output {
//         Scalar::new(format!(
//             "((unsigned char)((unsigned char){} + {}))",
//             self.val, rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Scalar<u8>) -> Self::Output {
//         Scalar::new(format!(
//             "((unsigned char)((unsigned char){} - {}))",
//             self.val, rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Scalar<u8>) -> Self::Output {
//         Scalar::new(format!(
//             "((unsigned char)((unsigned char){} * {}))",
//             self.val, rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Scalar<u8>) -> Self::Output {
//         Scalar::new(format!(
//             "(({} != 0) ? (unsigned char)((unsigned char){} % {}) : 0)",
//             rhs.val, self.val, rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Scalar<u8>) -> Self::Output {
//         Scalar::new(format!("max((unsigned char){}, {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Scalar<u8>) -> Self::Output {
//         Scalar::new(format!("min((unsigned char){}, {})", self.val, rhs.val))
//     }
// }
// impl NormalOut<Scalar<u16>> for Scalar<bool> {
//     type Output = Scalar<u16>;
//     #[inline(always)]
//     fn _pow(self, rhs: Scalar<u16>) -> Self::Output {
//         Scalar::new(format!(
//             "((unsigned short)powf((float)(unsigned short){}, (float){}))",
//             self.val, rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _clip(self, min: Scalar<u16>, max: Scalar<u16>) -> Self::Output {
//         // 先转换为 u16，然后用 min/max
//         Scalar::new(format!(
//             "min(max((unsigned short){}, {}), {})",
//             self.val, min.val, max.val
//         ))
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Scalar<u16>, b: Scalar<u16>) -> Self::Output {
//         // bool * a + b
//         Scalar::new(format!(
//             "((unsigned short)((unsigned short){} * {} + {}))",
//             self.val, a.val, b.val
//         ))
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Scalar<u16>) -> Self::Output {
//         Scalar::new(format!(
//             "((unsigned short)((unsigned short){} + {}))",
//             self.val, rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Scalar<u16>) -> Self::Output {
//         Scalar::new(format!(
//             "((unsigned short)((unsigned short){} - {}))",
//             self.val, rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Scalar<u16>) -> Self::Output {
//         Scalar::new(format!(
//             "((unsigned short)((unsigned short){} * {}))",
//             self.val, rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Scalar<u16>) -> Self::Output {
//         // 需要处理除数为0的情况
//         Scalar::new(format!(
//             "(({} != 0) ? (unsigned short)((unsigned short){} % {}) : 0)",
//             rhs.val, self.val, rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Scalar<u16>) -> Self::Output {
//         Scalar::new(format!("max((unsigned short){}, {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Scalar<u16>) -> Self::Output {
//         Scalar::new(format!("min((unsigned short){}, {})", self.val, rhs.val))
//     }
// }
// impl NormalOut<Scalar<u32>> for Scalar<bool> {
//     type Output = Scalar<u32>;
//     #[inline(always)]
//     fn _pow(self, rhs: Scalar<u32>) -> Self::Output {
//         Scalar::new(format!(
//             "((unsigned int)powf((float)(unsigned int){}, (float){}))",
//             self.val, rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _clip(self, min: Scalar<u32>, max: Scalar<u32>) -> Self::Output {
//         Scalar::new(format!(
//             "min(max((unsigned int){}, {}), {})",
//             self.val, min.val, max.val
//         ))
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Scalar<u32>, b: Scalar<u32>) -> Self::Output {
//         Scalar::new(format!(
//             "((unsigned int){} * {} + {})",
//             self.val, a.val, b.val
//         ))
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Scalar<u32>) -> Self::Output {
//         Scalar::new(format!("((unsigned int){} + {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Scalar<u32>) -> Self::Output {
//         Scalar::new(format!("((unsigned int){} - {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Scalar<u32>) -> Self::Output {
//         Scalar::new(format!("((unsigned int){} * {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Scalar<u32>) -> Self::Output {
//         // 需要处理除数为0的情况
//         Scalar::new(format!(
//             "({} != 0) ? ((unsigned int){} % {}) : 0",
//             rhs.val, self.val, rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Scalar<u32>) -> Self::Output {
//         Scalar::new(format!("max((unsigned int){}, {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Scalar<u32>) -> Self::Output {
//         Scalar::new(format!("min((unsigned int){}, {})", self.val, rhs.val))
//     }
// }
// impl NormalOut<Scalar<u64>> for Scalar<bool> {
//     type Output = Scalar<u64>;
//     #[inline(always)]
//     fn _pow(self, rhs: Scalar<u64>) -> Self::Output {
//         Scalar::new(format!(
//             "((unsigned long long)pow((double)(unsigned long long){}, (double){}))",
//             self.val, rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _clip(self, min: Scalar<u64>, max: Scalar<u64>) -> Self::Output {
//         Scalar::new(format!(
//             "min(max((unsigned long long){}, {}), {})",
//             self.val, min.val, max.val
//         ))
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Scalar<u64>, b: Scalar<u64>) -> Self::Output {
//         Scalar::new(format!(
//             "((unsigned long long){} * {} + {})",
//             self.val, a.val, b.val
//         ))
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Scalar<u64>) -> Self::Output {
//         Scalar::new(format!("((unsigned long long){} + {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Scalar<u64>) -> Self::Output {
//         Scalar::new(format!("((unsigned long long){} - {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Scalar<u64>) -> Self::Output {
//         Scalar::new(format!("((unsigned long long){} * {})", self.val, rhs.val))
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Scalar<u64>) -> Self::Output {
//         Scalar::new(format!(
//             "(({} != 0) ? ((unsigned long long){} % {}) : 0)",
//             rhs.val, self.val, rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Scalar<u64>) -> Self::Output {
//         Scalar::new(format!(
//             "max((unsigned long long){}, {})",
//             self.val, rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Scalar<u64>) -> Self::Output {
//         Scalar::new(format!(
//             "min((unsigned long long){}, {})",
//             self.val, rhs.val
//         ))
//     }
// }
// impl NormalOut<Scalar<half::bf16>> for Scalar<bool> {
//     type Output = Scalar<half::bf16>;
//     #[inline(always)]
//     fn _pow(self, rhs: Scalar<half::bf16>) -> Self::Output {
//         Scalar::new(format!(
//             "__float2bfloat16_rn(powf(__bfloat162float(({})), __bfloat162float({})))",
//             self.to_bf16().val,
//             rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _clip(self, min: Scalar<half::bf16>, max: Scalar<half::bf16>) -> Self::Output {
//         Scalar::new(format!(
//            "__float2bfloat16_rn(fminf(fmaxf(__bfloat162float(({})), __bfloat162float({})), __bfloat162float({})))",
//            self.to_bf16().val, min.val, max.val
//        ))
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Scalar<half::bf16>, b: Scalar<half::bf16>) -> Self::Output {
//         Scalar::new(format!(
//            "__float2bfloat16_rn(fmaf(__bfloat162float(({})), __bfloat162float({}), __bfloat162float({})))",
//            self.to_bf16().val, a.val, b.val
//        ))
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Scalar<half::bf16>) -> Self::Output {
//         Scalar::new(format!(
//             "__float2bfloat16_rn(__bfloat162float({}) + __bfloat162float({}))",
//             self.to_bf16().val,
//             rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Scalar<half::bf16>) -> Self::Output {
//         Scalar::new(format!(
//             "__float2bfloat16_rn(__bfloat162float({}) - __bfloat162float({}))",
//             self.to_bf16().val,
//             rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Scalar<half::bf16>) -> Self::Output {
//         Scalar::new(format!(
//             "__float2bfloat16_rn(__bfloat162float({}) * __bfloat162float({}))",
//             self.to_bf16().val,
//             rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Scalar<half::bf16>) -> Self::Output {
//         Scalar::new(format!(
//             "__float2bfloat16_rn(fmodf(__bfloat162float({}), __bfloat162float({})))",
//             self.to_bf16().val,
//             rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Scalar<half::bf16>) -> Self::Output {
//         Scalar::new(format!(
//             "__float2bfloat16_rn(fmaxf(__bfloat162float({}), __bfloat162float({})))",
//             self.to_bf16().val,
//             rhs.val
//         ))
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Scalar<half::bf16>) -> Self::Output {
//         Scalar::new(format!(
//             "__float2bfloat16_rn(fminf(__bfloat162float({}), __bfloat162float({})))",
//             self.to_bf16().val,
//             rhs.val
//         ))
//     }
// }
// impl NormalOut<Scalar<isize>> for Scalar<bool> {
//     type Output = Scalar<isize>;
//     #[inline(always)]
//     fn _pow(self, rhs: Scalar<isize>) -> Self::Output {
//         #[cfg(target_pointer_width = "64")]
//         return Scalar::new(format!(
//             "((long long)pow((double)(long long){}, (double){}))",
//             self.val, rhs.val
//         ));
//         #[cfg(target_pointer_width = "32")]
//         return Scalar::new(format!(
//             "((int)pow((double)(int){}, (double){}))",
//             self.val, rhs.val
//         ));
//     }
//     #[inline(always)]
//     fn _clip(self, min: Scalar<isize>, max: Scalar<isize>) -> Self::Output {
//         #[cfg(target_pointer_width = "64")]
//         return Scalar::new(format!(
//             "min(max((long long){}, {}), {})",
//             self.val, min.val, max.val
//         ));
//         #[cfg(target_pointer_width = "32")]
//         return Scalar::new(format!(
//             "min(max((int){}, {}), {})",
//             self.val, min.val, max.val
//         ));
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Scalar<isize>, b: Scalar<isize>) -> Self::Output {
//         #[cfg(target_pointer_width = "64")]
//         return Scalar::new(format!("((long long){} * {} + {})", self.val, a.val, b.val));
//         #[cfg(target_pointer_width = "32")]
//         return Scalar::new(format!("((int){} * {} + {})", self.val, a.val, b.val));
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Scalar<isize>) -> Self::Output {
//         #[cfg(target_pointer_width = "64")]
//         return Scalar::new(format!("((long long){} + {})", self.val, rhs.val));
//         #[cfg(target_pointer_width = "32")]
//         return Scalar::new(format!("((int){} + {})", self.val, rhs.val));
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Scalar<isize>) -> Self::Output {
//         #[cfg(target_pointer_width = "64")]
//         return Scalar::new(format!("((long long){} - {})", self.val, rhs.val));
//         #[cfg(target_pointer_width = "32")]
//         return Scalar::new(format!("((int){} - {})", self.val, rhs.val));
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Scalar<isize>) -> Self::Output {
//         #[cfg(target_pointer_width = "64")]
//         return Scalar::new(format!("((long long){} * {})", self.val, rhs.val));
//         #[cfg(target_pointer_width = "32")]
//         return Scalar::new(format!("((int){} * {})", self.val, rhs.val));
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Scalar<isize>) -> Self::Output {
//         #[cfg(target_pointer_width = "64")]
//         return Scalar::new(format!(
//             "(({} != 0) ? ((long long){} % {}) : 0)",
//             rhs.val, self.val, rhs.val
//         ));
//         #[cfg(target_pointer_width = "32")]
//         return Scalar::new(format!(
//             "(({} != 0) ? ((int){} % {}) : 0)",
//             rhs.val, self.val, rhs.val
//         ));
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Scalar<isize>) -> Self::Output {
//         #[cfg(target_pointer_width = "64")]
//         return Scalar::new(format!("max((long long){}, {})", self.val, rhs.val));
//         #[cfg(target_pointer_width = "32")]
//         return Scalar::new(format!("max((int){}, {})", self.val, rhs.val));
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Scalar<isize>) -> Self::Output {
//         #[cfg(target_pointer_width = "64")]
//         return Scalar::new(format!("min((long long){}, {})", self.val, rhs.val));
//         #[cfg(target_pointer_width = "32")]
//         return Scalar::new(format!("min((int){}, {})", self.val, rhs.val));
//     }
// }
// impl NormalOut<Scalar<usize>> for Scalar<bool> {
//     type Output = Scalar<usize>;
//     #[inline(always)]
//    fn _pow(self, rhs: Scalar<usize>) -> Self::Output {
//        Scalar::new(format!("((size_t)pow((double)(size_t){}, (double){}))", 
//            self.val, rhs.val))
//    }
//     #[inline(always)]
//    fn _clip(self, min: Scalar<usize>, max: Scalar<usize>) -> Self::Output {
//        Scalar::new(format!(
//            "min(max((size_t){}, {}), {})",
//            self.val, min.val, max.val
//        ))
//    }
//     #[inline(always)]
//    fn _mul_add(self, a: Scalar<usize>, b: Scalar<usize>) -> Self::Output {
//        Scalar::new(format!("((size_t){} * {} + {})", 
//            self.val, a.val, b.val))
//    }
//     #[inline(always)]
//    fn _add(self, rhs: Scalar<usize>) -> Self::Output {
//        Scalar::new(format!("((size_t){} + {})", 
//            self.val, rhs.val))
//    }
//     #[inline(always)]
//    fn _sub(self, rhs: Scalar<usize>) -> Self::Output {
//        Scalar::new(format!("((size_t){} - {})", 
//            self.val, rhs.val))
//    }
//     #[inline(always)]
//    fn _mul(self, rhs: Scalar<usize>) -> Self::Output {
//        Scalar::new(format!("((size_t){} * {})", 
//            self.val, rhs.val))
//    }
//     #[inline(always)]
//    fn _rem(self, rhs: Scalar<usize>) -> Self::Output {
//        Scalar::new(format!("(({} != 0) ? ((size_t){} % {}) : 0)", 
//            rhs.val, self.val, rhs.val))
//    }
//     #[inline(always)]
//    fn _max(self, rhs: Scalar<usize>) -> Self::Output {
//        Scalar::new(format!("max((size_t){}, {})", 
//            self.val, rhs.val))
//    }
//     #[inline(always)]
//    fn _min(self, rhs: Scalar<usize>) -> Self::Output {
//        Scalar::new(format!("min((size_t){}, {})", 
//            self.val, rhs.val))
//    }
// }
// impl NormalOut<Scalar<Complex32>> for Scalar<bool> {
//     type Output = Scalar<Complex32>;
//     #[inline(always)]
//    fn _pow(self, rhs: Scalar<Complex32>) -> Self::Output {
//        Scalar::new(format!(
//            "cuCpowf(make_cuComplex((float){}, 0.0f), {})",
//            self.val, rhs.val
//        ))
//    }
//     #[inline(always)]
//    fn _clip(self, min: Scalar<Complex32>, max: Scalar<Complex32>) -> Self::Output {
//        Scalar::new(format!("make_cuComplex((float){}, 0.0f)", self.val))
//    }
//     #[inline(always)]
//    fn _mul_add(self, a: Scalar<Complex32>, b: Scalar<Complex32>) -> Self::Output {
//        Scalar::new(format!(
//            "cuCaddf(cuCmulf(make_cuComplex((float){}, 0.0f), {}), {})",
//            self.val, a.val, b.val
//        ))
//    }
//     #[inline(always)]
//    fn _add(self, rhs: Scalar<Complex32>) -> Self::Output {
//        Scalar::new(format!(
//            "cuCaddf(make_cuComplex((float){}, 0.0f), {})",
//            self.val, rhs.val
//        ))
//    }
//     #[inline(always)]
//    fn _sub(self, rhs: Scalar<Complex32>) -> Self::Output {
//        Scalar::new(format!(
//            "cuCsubf(make_cuComplex((float){}, 0.0f), {})",
//            self.val, rhs.val
//        ))
//    }
//     #[inline(always)]
//    fn _mul(self, rhs: Scalar<Complex32>) -> Self::Output {
//        Scalar::new(format!(
//            "cuCmulf(make_cuComplex((float){}, 0.0f), {})",
//            self.val, rhs.val
//        ))
//    }
//     #[inline(always)]
//    fn _rem(self, rhs: Scalar<Complex32>) -> Self::Output {
//        Scalar::new("make_cuComplex(0.0f, 0.0f)".to_string())
//    }
//     #[inline(always)]
//    fn _max(self, rhs: Scalar<Complex32>) -> Self::Output {
//        Scalar::new(format!(
//            "cuCabsf(make_cuComplex((float){}, 0.0f)) > cuCabsf({}) ? make_cuComplex((float){}, 0.0f) : {}",
//            self.val, rhs.val, self.val, rhs.val
//        ))
//    }
//     #[inline(always)]
//    fn _min(self, rhs: Scalar<Complex32>) -> Self::Output {
//        Scalar::new(format!(
//            "cuCabsf(make_cuComplex((float){}, 0.0f)) < cuCabsf({}) ? make_cuComplex((float){}, 0.0f) : {}",
//            self.val, rhs.val, self.val, rhs.val
//        ))
//    }
// }
// impl NormalOut<Scalar<Complex64>> for Scalar<bool> {
//     type Output = Scalar<Complex64>;
//     #[inline(always)]
//    fn _pow(self, rhs: Scalar<Complex64>) -> Self::Output {
//        Scalar::new(format!(
//            "cuCpow(make_cuDoubleComplex((double){}, 0.0), {})",
//            self.val, rhs.val
//        ))
//    }
//     #[inline(always)]
//    fn _clip(self, min: Scalar<Complex64>, max: Scalar<Complex64>) -> Self::Output {
//        Scalar::new(format!("make_cuDoubleComplex((double){}, 0.0)", self.val))
//    }
//     #[inline(always)]
//    fn _mul_add(self, a: Scalar<Complex64>, b: Scalar<Complex64>) -> Self::Output {
//        Scalar::new(format!(
//            "cuCadd(cuCmul(make_cuDoubleComplex((double){}, 0.0), {}), {})",
//            self.val, a.val, b.val
//        ))
//    }
//     #[inline(always)]
//    fn _add(self, rhs: Scalar<Complex64>) -> Self::Output {
//        Scalar::new(format!(
//            "cuCadd(make_cuDoubleComplex((double){}, 0.0), {})",
//            self.val, rhs.val
//        ))
//    }
//     #[inline(always)]
//    fn _sub(self, rhs: Scalar<Complex64>) -> Self::Output {
//        Scalar::new(format!(
//            "cuCsub(make_cuDoubleComplex((double){}, 0.0), {})",
//            self.val, rhs.val
//        ))
//    }
//     #[inline(always)]
//    fn _mul(self, rhs: Scalar<Complex64>) -> Self::Output {
//        Scalar::new(format!(
//            "cuCmul(make_cuDoubleComplex((double){}, 0.0), {})",
//            self.val, rhs.val
//        ))
//    }
//     #[inline(always)]
//    fn _rem(self, rhs: Scalar<Complex64>) -> Self::Output {
//        Scalar::new("make_cuDoubleComplex(0.0, 0.0)".to_string())
//    }
//     #[inline(always)]
//    fn _max(self, rhs: Scalar<Complex64>) -> Self::Output {
//        Scalar::new(format!(
//            "cuCabs(make_cuDoubleComplex((double){}, 0.0)) > cuCabs({}) ? make_cuDoubleComplex((double){}, 0.0) : {}",
//            self.val, rhs.val, self.val, rhs.val
//        ))
//    }
//     #[inline(always)]
//    fn _min(self, rhs: Scalar<Complex64>) -> Self::Output {
//        Scalar::new(format!(
//            "cuCabs(make_cuDoubleComplex((double){}, 0.0)) < cuCabs({}) ? make_cuDoubleComplex((double){}, 0.0) : {}",
//            self.val, rhs.val, self.val, rhs.val
//        ))
//    }
// }
// impl NormalOut<bool> for half::f16 {
//     type Output = half::f16;
//     #[inline(always)]
//     fn _pow(self, rhs: bool) -> Self::Output {
//         self.to_f16().powf(rhs.to_f16())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bool, max: bool) -> Self::Output {
//         paste::paste! {
//             let a = self.to_f16();
//             let min = min.to_f16();
//             let max = max.to_f16();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bool, b: bool) -> Self::Output {
//         self.to_f16() * a.to_f16() + b.to_f16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bool) -> Self::Output {
//         self.to_f16() + rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bool) -> Self::Output {
//         self.to_f16() - rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bool) -> Self::Output {
//         self.to_f16() * rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bool) -> Self::Output {
//         self.to_f16() % rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bool) -> Self::Output {
//         self.to_f16().max(rhs.to_f16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bool) -> Self::Output {
//         self.to_f16().min(rhs.to_f16())
//     }
// }
// impl NormalOut<half::f16> for half::f16 {
//     type Output = half::f16;
//     #[inline(always)]
//     fn _pow(self, rhs: half::f16) -> Self::Output {
//         self.to_f16().powf(rhs.to_f16())
//     }
//     #[inline(always)]
//     fn _clip(self, min: half::f16, max: half::f16) -> Self::Output {
//         paste::paste! {
//             let a = self.to_f16();
//             let min = min.to_f16();
//             let max = max.to_f16();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: half::f16, b: half::f16) -> Self::Output {
//         self.to_f16() * a.to_f16() + b.to_f16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: half::f16) -> Self::Output {
//         self.to_f16() + rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: half::f16) -> Self::Output {
//         self.to_f16() - rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: half::f16) -> Self::Output {
//         self.to_f16() * rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: half::f16) -> Self::Output {
//         self.to_f16() % rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: half::f16) -> Self::Output {
//         self.to_f16().max(rhs.to_f16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: half::f16) -> Self::Output {
//         self.to_f16().min(rhs.to_f16())
//     }
// }
// impl NormalOut<f32> for half::f16 {
//     type Output = f32;
//     #[inline(always)]
//     fn _pow(self, rhs: f32) -> Self::Output {
//         self.to_f32().powf(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f32, max: f32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f32>]();
//             let min = min.[<to_ f32>]();
//             let max = max.[<to_ f32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f32, b: f32) -> Self::Output {
//         self.to_f32() * a.to_f32() + b.to_f32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f32) -> Self::Output {
//         self.to_f32() + rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f32) -> Self::Output {
//         self.to_f32() - rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f32) -> Self::Output {
//         self.to_f32() * rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f32) -> Self::Output {
//         self.to_f32() % rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f32) -> Self::Output {
//         self.to_f32().max(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f32) -> Self::Output {
//         self.to_f32().min(rhs.to_f32())
//     }
// }
// impl NormalOut<f64> for half::f16 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: f64) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f64, max: f64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f64, b: f64) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f64) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f64) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f64) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f64) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f64) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f64) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<i8> for half::f16 {
//     type Output = half::f16;
//     #[inline(always)]
//     fn _pow(self, rhs: i8) -> Self::Output {
//         self.to_f16().powf(rhs.to_f16())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i8, max: i8) -> Self::Output {
//         paste::paste! {
//             let a = self.to_f16();
//             let min = min.to_f16();
//             let max = max.to_f16();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i8, b: i8) -> Self::Output {
//         self.to_f16() * a.to_f16() + b.to_f16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i8) -> Self::Output {
//         self.to_f16() + rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i8) -> Self::Output {
//         self.to_f16() - rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i8) -> Self::Output {
//         self.to_f16() * rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i8) -> Self::Output {
//         self.to_f16() % rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i8) -> Self::Output {
//         self.to_f16().max(rhs.to_f16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i8) -> Self::Output {
//         self.to_f16().min(rhs.to_f16())
//     }
// }
// impl NormalOut<i16> for half::f16 {
//     type Output = half::f16;
//     #[inline(always)]
//     fn _pow(self, rhs: i16) -> Self::Output {
//         self.to_f16().powf(rhs.to_f16())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i16, max: i16) -> Self::Output {
//         paste::paste! {
//             let a = self.to_f16();
//             let min = min.to_f16();
//             let max = max.to_f16();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i16, b: i16) -> Self::Output {
//         self.to_f16() * a.to_f16() + b.to_f16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i16) -> Self::Output {
//         self.to_f16() + rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i16) -> Self::Output {
//         self.to_f16() - rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i16) -> Self::Output {
//         self.to_f16() * rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i16) -> Self::Output {
//         self.to_f16() % rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i16) -> Self::Output {
//         self.to_f16().max(rhs.to_f16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i16) -> Self::Output {
//         self.to_f16().min(rhs.to_f16())
//     }
// }
// impl NormalOut<i32> for half::f16 {
//     type Output = f32;
//     #[inline(always)]
//     fn _pow(self, rhs: i32) -> Self::Output {
//         self.to_f32().powf(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i32, max: i32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f32>]();
//             let min = min.[<to_ f32>]();
//             let max = max.[<to_ f32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i32, b: i32) -> Self::Output {
//         self.to_f32() * a.to_f32() + b.to_f32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i32) -> Self::Output {
//         self.to_f32() + rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i32) -> Self::Output {
//         self.to_f32() - rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i32) -> Self::Output {
//         self.to_f32() * rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i32) -> Self::Output {
//         self.to_f32() % rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i32) -> Self::Output {
//         self.to_f32().max(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i32) -> Self::Output {
//         self.to_f32().min(rhs.to_f32())
//     }
// }
// impl NormalOut<i64> for half::f16 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: i64) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i64, max: i64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i64, b: i64) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i64) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i64) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i64) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i64) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i64) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i64) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<u8> for half::f16 {
//     type Output = half::f16;
//     #[inline(always)]
//     fn _pow(self, rhs: u8) -> Self::Output {
//         self.to_f16().powf(rhs.to_f16())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u8, max: u8) -> Self::Output {
//         paste::paste! {
//             let a = self.to_f16();
//             let min = min.to_f16();
//             let max = max.to_f16();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u8, b: u8) -> Self::Output {
//         self.to_f16() * a.to_f16() + b.to_f16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u8) -> Self::Output {
//         self.to_f16() + rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u8) -> Self::Output {
//         self.to_f16() - rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u8) -> Self::Output {
//         self.to_f16() * rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u8) -> Self::Output {
//         self.to_f16() % rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u8) -> Self::Output {
//         self.to_f16().max(rhs.to_f16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u8) -> Self::Output {
//         self.to_f16().min(rhs.to_f16())
//     }
// }
// impl NormalOut<u16> for half::f16 {
//     type Output = f32;
//     #[inline(always)]
//     fn _pow(self, rhs: u16) -> Self::Output {
//         self.to_f32().powf(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u16, max: u16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f32>]();
//             let min = min.[<to_ f32>]();
//             let max = max.[<to_ f32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u16, b: u16) -> Self::Output {
//         self.to_f32() * a.to_f32() + b.to_f32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u16) -> Self::Output {
//         self.to_f32() + rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u16) -> Self::Output {
//         self.to_f32() - rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u16) -> Self::Output {
//         self.to_f32() * rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u16) -> Self::Output {
//         self.to_f32() % rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u16) -> Self::Output {
//         self.to_f32().max(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u16) -> Self::Output {
//         self.to_f32().min(rhs.to_f32())
//     }
// }
// impl NormalOut<u32> for half::f16 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: u32) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u32, max: u32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u32, b: u32) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u32) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u32) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u32) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u32) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u32) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u32) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<u64> for half::f16 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: u64) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u64, max: u64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u64, b: u64) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u64) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u64) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u64) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u64) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u64) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u64) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<bf16> for half::f16 {
//     type Output = bf16;
//     #[inline(always)]
//     fn _pow(self, rhs: bf16) -> Self::Output {
//         self.to_bf16().powf(rhs.to_bf16())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bf16, max: bf16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ bf16>]();
//             let min = min.[<to_ bf16>]();
//             let max = max.[<to_ bf16>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bf16, b: bf16) -> Self::Output {
//         self.to_bf16() * a.to_bf16() + b.to_bf16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bf16) -> Self::Output {
//         self.to_bf16() + rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bf16) -> Self::Output {
//         self.to_bf16() - rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bf16) -> Self::Output {
//         self.to_bf16() * rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bf16) -> Self::Output {
//         self.to_bf16() % rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bf16) -> Self::Output {
//         self.to_bf16().max(rhs.to_bf16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bf16) -> Self::Output {
//         self.to_bf16().min(rhs.to_bf16())
//     }
// }
// impl NormalOut<isize> for half::f16 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: isize) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: isize, max: isize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: isize, b: isize) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: isize) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: isize) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: isize) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: isize) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: isize) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: isize) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<usize> for half::f16 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: usize) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: usize, max: usize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: usize, b: usize) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: usize) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: usize) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: usize) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: usize) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: usize) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: usize) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<Complex32> for half::f16 {
//     type Output = Complex32;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32().powc(rhs.to_complex32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex32, max: Complex32) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex32>]();
//             let min = min.[<to_ complex32>]();
//             let max = max.[<to_ complex32>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex32::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex32, b: Complex32) -> Self::Output {
//         self.to_complex32() * a.to_complex32() + b.to_complex32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() + rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() - rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() * rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() % rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex32) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex32) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<Complex64> for half::f16 {
//     type Output = Complex64;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64().powc(rhs.to_complex64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex64, max: Complex64) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex64>]();
//             let min = min.[<to_ complex64>]();
//             let max = max.[<to_ complex64>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex64::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex64, b: Complex64) -> Self::Output {
//         self.to_complex64() * a.to_complex64() + b.to_complex64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() + rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() - rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() * rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() % rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex64) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex64) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<bool> for f32 {
//     type Output = f32;
//     #[inline(always)]
//     fn _pow(self, rhs: bool) -> Self::Output {
//         self.to_f32().powf(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bool, max: bool) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f32>]();
//             let min = min.[<to_ f32>]();
//             let max = max.[<to_ f32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bool, b: bool) -> Self::Output {
//         self.to_f32() * a.to_f32() + b.to_f32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bool) -> Self::Output {
//         self.to_f32() + rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bool) -> Self::Output {
//         self.to_f32() - rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bool) -> Self::Output {
//         self.to_f32() * rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bool) -> Self::Output {
//         self.to_f32() % rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bool) -> Self::Output {
//         self.to_f32().max(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bool) -> Self::Output {
//         self.to_f32().min(rhs.to_f32())
//     }
// }
// impl NormalOut<half::f16> for f32 {
//     type Output = f32;
//     #[inline(always)]
//     fn _pow(self, rhs: half::f16) -> Self::Output {
//         self.to_f32().powf(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: half::f16, max: half::f16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f32>]();
//             let min = min.[<to_ f32>]();
//             let max = max.[<to_ f32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: half::f16, b: half::f16) -> Self::Output {
//         self.to_f32() * a.to_f32() + b.to_f32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: half::f16) -> Self::Output {
//         self.to_f32() + rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: half::f16) -> Self::Output {
//         self.to_f32() - rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: half::f16) -> Self::Output {
//         self.to_f32() * rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: half::f16) -> Self::Output {
//         self.to_f32() % rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: half::f16) -> Self::Output {
//         self.to_f32().max(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: half::f16) -> Self::Output {
//         self.to_f32().min(rhs.to_f32())
//     }
// }
// impl NormalOut<f32> for f32 {
//     type Output = f32;
//     #[inline(always)]
//     fn _pow(self, rhs: f32) -> Self::Output {
//         self.to_f32().powf(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f32, max: f32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f32>]();
//             let min = min.[<to_ f32>]();
//             let max = max.[<to_ f32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f32, b: f32) -> Self::Output {
//         self.to_f32() * a.to_f32() + b.to_f32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f32) -> Self::Output {
//         self.to_f32() + rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f32) -> Self::Output {
//         self.to_f32() - rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f32) -> Self::Output {
//         self.to_f32() * rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f32) -> Self::Output {
//         self.to_f32() % rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f32) -> Self::Output {
//         self.to_f32().max(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f32) -> Self::Output {
//         self.to_f32().min(rhs.to_f32())
//     }
// }
// impl NormalOut<f64> for f32 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: f64) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f64, max: f64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f64, b: f64) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f64) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f64) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f64) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f64) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f64) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f64) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<i8> for f32 {
//     type Output = f32;
//     #[inline(always)]
//     fn _pow(self, rhs: i8) -> Self::Output {
//         self.to_f32().powf(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i8, max: i8) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f32>]();
//             let min = min.[<to_ f32>]();
//             let max = max.[<to_ f32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i8, b: i8) -> Self::Output {
//         self.to_f32() * a.to_f32() + b.to_f32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i8) -> Self::Output {
//         self.to_f32() + rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i8) -> Self::Output {
//         self.to_f32() - rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i8) -> Self::Output {
//         self.to_f32() * rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i8) -> Self::Output {
//         self.to_f32() % rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i8) -> Self::Output {
//         self.to_f32().max(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i8) -> Self::Output {
//         self.to_f32().min(rhs.to_f32())
//     }
// }
// impl NormalOut<i16> for f32 {
//     type Output = f32;
//     #[inline(always)]
//     fn _pow(self, rhs: i16) -> Self::Output {
//         self.to_f32().powf(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i16, max: i16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f32>]();
//             let min = min.[<to_ f32>]();
//             let max = max.[<to_ f32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i16, b: i16) -> Self::Output {
//         self.to_f32() * a.to_f32() + b.to_f32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i16) -> Self::Output {
//         self.to_f32() + rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i16) -> Self::Output {
//         self.to_f32() - rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i16) -> Self::Output {
//         self.to_f32() * rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i16) -> Self::Output {
//         self.to_f32() % rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i16) -> Self::Output {
//         self.to_f32().max(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i16) -> Self::Output {
//         self.to_f32().min(rhs.to_f32())
//     }
// }
// impl NormalOut<i32> for f32 {
//     type Output = f32;
//     #[inline(always)]
//     fn _pow(self, rhs: i32) -> Self::Output {
//         self.to_f32().powf(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i32, max: i32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f32>]();
//             let min = min.[<to_ f32>]();
//             let max = max.[<to_ f32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i32, b: i32) -> Self::Output {
//         self.to_f32() * a.to_f32() + b.to_f32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i32) -> Self::Output {
//         self.to_f32() + rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i32) -> Self::Output {
//         self.to_f32() - rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i32) -> Self::Output {
//         self.to_f32() * rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i32) -> Self::Output {
//         self.to_f32() % rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i32) -> Self::Output {
//         self.to_f32().max(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i32) -> Self::Output {
//         self.to_f32().min(rhs.to_f32())
//     }
// }
// impl NormalOut<i64> for f32 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: i64) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i64, max: i64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i64, b: i64) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i64) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i64) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i64) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i64) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i64) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i64) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<u8> for f32 {
//     type Output = f32;
//     #[inline(always)]
//     fn _pow(self, rhs: u8) -> Self::Output {
//         self.to_f32().powf(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u8, max: u8) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f32>]();
//             let min = min.[<to_ f32>]();
//             let max = max.[<to_ f32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u8, b: u8) -> Self::Output {
//         self.to_f32() * a.to_f32() + b.to_f32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u8) -> Self::Output {
//         self.to_f32() + rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u8) -> Self::Output {
//         self.to_f32() - rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u8) -> Self::Output {
//         self.to_f32() * rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u8) -> Self::Output {
//         self.to_f32() % rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u8) -> Self::Output {
//         self.to_f32().max(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u8) -> Self::Output {
//         self.to_f32().min(rhs.to_f32())
//     }
// }
// impl NormalOut<u16> for f32 {
//     type Output = f32;
//     #[inline(always)]
//     fn _pow(self, rhs: u16) -> Self::Output {
//         self.to_f32().powf(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u16, max: u16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f32>]();
//             let min = min.[<to_ f32>]();
//             let max = max.[<to_ f32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u16, b: u16) -> Self::Output {
//         self.to_f32() * a.to_f32() + b.to_f32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u16) -> Self::Output {
//         self.to_f32() + rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u16) -> Self::Output {
//         self.to_f32() - rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u16) -> Self::Output {
//         self.to_f32() * rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u16) -> Self::Output {
//         self.to_f32() % rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u16) -> Self::Output {
//         self.to_f32().max(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u16) -> Self::Output {
//         self.to_f32().min(rhs.to_f32())
//     }
// }
// impl NormalOut<u32> for f32 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: u32) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u32, max: u32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u32, b: u32) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u32) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u32) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u32) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u32) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u32) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u32) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<u64> for f32 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: u64) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u64, max: u64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u64, b: u64) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u64) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u64) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u64) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u64) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u64) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u64) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<bf16> for f32 {
//     type Output = f32;
//     #[inline(always)]
//     fn _pow(self, rhs: bf16) -> Self::Output {
//         self.to_f32().powf(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bf16, max: bf16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f32>]();
//             let min = min.[<to_ f32>]();
//             let max = max.[<to_ f32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bf16, b: bf16) -> Self::Output {
//         self.to_f32() * a.to_f32() + b.to_f32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bf16) -> Self::Output {
//         self.to_f32() + rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bf16) -> Self::Output {
//         self.to_f32() - rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bf16) -> Self::Output {
//         self.to_f32() * rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bf16) -> Self::Output {
//         self.to_f32() % rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bf16) -> Self::Output {
//         self.to_f32().max(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bf16) -> Self::Output {
//         self.to_f32().min(rhs.to_f32())
//     }
// }
// impl NormalOut<isize> for f32 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: isize) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: isize, max: isize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: isize, b: isize) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: isize) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: isize) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: isize) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: isize) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: isize) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: isize) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<usize> for f32 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: usize) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: usize, max: usize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: usize, b: usize) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: usize) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: usize) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: usize) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: usize) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: usize) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: usize) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<Complex32> for f32 {
//     type Output = Complex32;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32().powc(rhs.to_complex32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex32, max: Complex32) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex32>]();
//             let min = min.[<to_ complex32>]();
//             let max = max.[<to_ complex32>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex32::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex32, b: Complex32) -> Self::Output {
//         self.to_complex32() * a.to_complex32() + b.to_complex32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() + rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() - rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() * rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() % rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex32) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex32) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<Complex64> for f32 {
//     type Output = Complex64;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64().powc(rhs.to_complex64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex64, max: Complex64) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex64>]();
//             let min = min.[<to_ complex64>]();
//             let max = max.[<to_ complex64>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex64::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex64, b: Complex64) -> Self::Output {
//         self.to_complex64() * a.to_complex64() + b.to_complex64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() + rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() - rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() * rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() % rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex64) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex64) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<bool> for f64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: bool) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bool, max: bool) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bool, b: bool) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bool) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bool) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bool) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bool) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bool) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bool) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<half::f16> for f64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: half::f16) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: half::f16, max: half::f16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: half::f16, b: half::f16) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: half::f16) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: half::f16) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: half::f16) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: half::f16) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: half::f16) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: half::f16) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<f32> for f64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: f32) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f32, max: f32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f32, b: f32) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f32) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f32) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f32) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f32) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f32) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f32) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<f64> for f64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: f64) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f64, max: f64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f64, b: f64) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f64) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f64) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f64) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f64) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f64) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f64) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<i8> for f64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: i8) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i8, max: i8) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i8, b: i8) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i8) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i8) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i8) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i8) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i8) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i8) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<i16> for f64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: i16) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i16, max: i16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i16, b: i16) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i16) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i16) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i16) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i16) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i16) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i16) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<i32> for f64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: i32) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i32, max: i32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i32, b: i32) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i32) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i32) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i32) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i32) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i32) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i32) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<i64> for f64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: i64) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i64, max: i64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i64, b: i64) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i64) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i64) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i64) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i64) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i64) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i64) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<u8> for f64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: u8) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u8, max: u8) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u8, b: u8) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u8) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u8) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u8) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u8) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u8) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u8) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<u16> for f64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: u16) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u16, max: u16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u16, b: u16) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u16) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u16) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u16) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u16) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u16) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u16) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<u32> for f64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: u32) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u32, max: u32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u32, b: u32) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u32) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u32) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u32) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u32) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u32) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u32) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<u64> for f64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: u64) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u64, max: u64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u64, b: u64) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u64) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u64) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u64) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u64) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u64) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u64) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<bf16> for f64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: bf16) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bf16, max: bf16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bf16, b: bf16) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bf16) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bf16) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bf16) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bf16) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bf16) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bf16) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<isize> for f64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: isize) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: isize, max: isize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: isize, b: isize) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: isize) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: isize) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: isize) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: isize) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: isize) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: isize) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<usize> for f64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: usize) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: usize, max: usize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: usize, b: usize) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: usize) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: usize) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: usize) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: usize) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: usize) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: usize) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<Complex32> for f64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex32) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex32, max: Complex32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex32, b: Complex32) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex32) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex32) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex32) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex32) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex32) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex32) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<Complex64> for f64 {
//     type Output = Complex64;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64().powc(rhs.to_complex64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex64, max: Complex64) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex64>]();
//             let min = min.[<to_ complex64>]();
//             let max = max.[<to_ complex64>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex64::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex64, b: Complex64) -> Self::Output {
//         self.to_complex64() * a.to_complex64() + b.to_complex64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() + rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() - rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() * rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() % rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex64) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex64) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<bool> for i8 {
//     type Output = i8;
//     #[inline(always)]
//     fn _pow(self, rhs: bool) -> Self::Output {
//         self.to_i8().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bool, max: bool) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i8>]();
//             let min = min.[<to_ i8>]();
//             let max = max.[<to_ i8>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bool, b: bool) -> Self::Output {
//         self.to_i8().wrapping_mul(a.to_i8()) + b.to_i8()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bool) -> Self::Output {
//         self.to_i8().wrapping_add(rhs.to_i8())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bool) -> Self::Output {
//         self.to_i8().wrapping_sub(rhs.to_i8())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bool) -> Self::Output {
//         self.to_i8().wrapping_mul(rhs.to_i8())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bool) -> Self::Output {
//         self.to_i8().wrapping_rem(rhs.to_i8())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bool) -> Self::Output {
//         self.to_i8().max(rhs.to_i8())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bool) -> Self::Output {
//         self.to_i8().min(rhs.to_i8())
//     }
// }
// impl NormalOut<half::f16> for i8 {
//     type Output = half::f16;
//     #[inline(always)]
//     fn _pow(self, rhs: half::f16) -> Self::Output {
//         self.to_f16().powf(rhs.to_f16())
//     }
//     #[inline(always)]
//     fn _clip(self, min: half::f16, max: half::f16) -> Self::Output {
//         paste::paste! {
//             let a = self.to_f16();
//             let min = min.to_f16();
//             let max = max.to_f16();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: half::f16, b: half::f16) -> Self::Output {
//         self.to_f16() * a.to_f16() + b.to_f16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: half::f16) -> Self::Output {
//         self.to_f16() + rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: half::f16) -> Self::Output {
//         self.to_f16() - rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: half::f16) -> Self::Output {
//         self.to_f16() * rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: half::f16) -> Self::Output {
//         self.to_f16() % rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: half::f16) -> Self::Output {
//         self.to_f16().max(rhs.to_f16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: half::f16) -> Self::Output {
//         self.to_f16().min(rhs.to_f16())
//     }
// }
// impl NormalOut<f32> for i8 {
//     type Output = f32;
//     #[inline(always)]
//     fn _pow(self, rhs: f32) -> Self::Output {
//         self.to_f32().powf(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f32, max: f32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f32>]();
//             let min = min.[<to_ f32>]();
//             let max = max.[<to_ f32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f32, b: f32) -> Self::Output {
//         self.to_f32() * a.to_f32() + b.to_f32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f32) -> Self::Output {
//         self.to_f32() + rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f32) -> Self::Output {
//         self.to_f32() - rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f32) -> Self::Output {
//         self.to_f32() * rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f32) -> Self::Output {
//         self.to_f32() % rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f32) -> Self::Output {
//         self.to_f32().max(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f32) -> Self::Output {
//         self.to_f32().min(rhs.to_f32())
//     }
// }
// impl NormalOut<f64> for i8 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: f64) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f64, max: f64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f64, b: f64) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f64) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f64) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f64) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f64) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f64) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f64) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<i8> for i8 {
//     type Output = i8;
//     #[inline(always)]
//     fn _pow(self, rhs: i8) -> Self::Output {
//         self.to_i8().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i8, max: i8) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i8>]();
//             let min = min.[<to_ i8>]();
//             let max = max.[<to_ i8>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i8, b: i8) -> Self::Output {
//         self.to_i8().wrapping_mul(a.to_i8()) + b.to_i8()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i8) -> Self::Output {
//         self.to_i8().wrapping_add(rhs.to_i8())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i8) -> Self::Output {
//         self.to_i8().wrapping_sub(rhs.to_i8())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i8) -> Self::Output {
//         self.to_i8().wrapping_mul(rhs.to_i8())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i8) -> Self::Output {
//         self.to_i8().wrapping_rem(rhs.to_i8())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i8) -> Self::Output {
//         self.to_i8().max(rhs.to_i8())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i8) -> Self::Output {
//         self.to_i8().min(rhs.to_i8())
//     }
// }
// impl NormalOut<i16> for i8 {
//     type Output = i16;
//     #[inline(always)]
//     fn _pow(self, rhs: i16) -> Self::Output {
//         self.to_i16().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i16, max: i16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i16>]();
//             let min = min.[<to_ i16>]();
//             let max = max.[<to_ i16>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i16, b: i16) -> Self::Output {
//         self.to_i16().wrapping_mul(a.to_i16()) + b.to_i16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i16) -> Self::Output {
//         self.to_i16().wrapping_add(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i16) -> Self::Output {
//         self.to_i16().wrapping_sub(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i16) -> Self::Output {
//         self.to_i16().wrapping_mul(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i16) -> Self::Output {
//         self.to_i16().wrapping_rem(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i16) -> Self::Output {
//         self.to_i16().max(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i16) -> Self::Output {
//         self.to_i16().min(rhs.to_i16())
//     }
// }
// impl NormalOut<i32> for i8 {
//     type Output = i32;
//     #[inline(always)]
//     fn _pow(self, rhs: i32) -> Self::Output {
//         self.to_i32().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i32, max: i32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i32>]();
//             let min = min.[<to_ i32>]();
//             let max = max.[<to_ i32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i32, b: i32) -> Self::Output {
//         self.to_i32().wrapping_mul(a.to_i32()) + b.to_i32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i32) -> Self::Output {
//         self.to_i32().wrapping_add(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i32) -> Self::Output {
//         self.to_i32().wrapping_sub(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i32) -> Self::Output {
//         self.to_i32().wrapping_mul(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i32) -> Self::Output {
//         self.to_i32().wrapping_rem(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i32) -> Self::Output {
//         self.to_i32().max(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i32) -> Self::Output {
//         self.to_i32().min(rhs.to_i32())
//     }
// }
// impl NormalOut<i64> for i8 {
//     type Output = i64;
//     #[inline(always)]
//     fn _pow(self, rhs: i64) -> Self::Output {
//         self.to_i64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i64, max: i64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i64>]();
//             let min = min.[<to_ i64>]();
//             let max = max.[<to_ i64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i64, b: i64) -> Self::Output {
//         self.to_i64().wrapping_mul(a.to_i64()) + b.to_i64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_add(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_sub(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_mul(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_rem(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i64) -> Self::Output {
//         self.to_i64().max(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i64) -> Self::Output {
//         self.to_i64().min(rhs.to_i64())
//     }
// }
// impl NormalOut<u8> for i8 {
//     type Output = i16;
//     #[inline(always)]
//     fn _pow(self, rhs: u8) -> Self::Output {
//         self.to_i16().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u8, max: u8) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i16>]();
//             let min = min.[<to_ i16>]();
//             let max = max.[<to_ i16>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u8, b: u8) -> Self::Output {
//         self.to_i16().wrapping_mul(a.to_i16()) + b.to_i16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u8) -> Self::Output {
//         self.to_i16().wrapping_add(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u8) -> Self::Output {
//         self.to_i16().wrapping_sub(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u8) -> Self::Output {
//         self.to_i16().wrapping_mul(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u8) -> Self::Output {
//         self.to_i16().wrapping_rem(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u8) -> Self::Output {
//         self.to_i16().max(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u8) -> Self::Output {
//         self.to_i16().min(rhs.to_i16())
//     }
// }
// impl NormalOut<u16> for i8 {
//     type Output = i32;
//     #[inline(always)]
//     fn _pow(self, rhs: u16) -> Self::Output {
//         self.to_i32().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u16, max: u16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i32>]();
//             let min = min.[<to_ i32>]();
//             let max = max.[<to_ i32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u16, b: u16) -> Self::Output {
//         self.to_i32().wrapping_mul(a.to_i32()) + b.to_i32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u16) -> Self::Output {
//         self.to_i32().wrapping_add(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u16) -> Self::Output {
//         self.to_i32().wrapping_sub(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u16) -> Self::Output {
//         self.to_i32().wrapping_mul(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u16) -> Self::Output {
//         self.to_i32().wrapping_rem(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u16) -> Self::Output {
//         self.to_i32().max(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u16) -> Self::Output {
//         self.to_i32().min(rhs.to_i32())
//     }
// }
// impl NormalOut<u32> for i8 {
//     type Output = i64;
//     #[inline(always)]
//     fn _pow(self, rhs: u32) -> Self::Output {
//         self.to_i64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u32, max: u32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i64>]();
//             let min = min.[<to_ i64>]();
//             let max = max.[<to_ i64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u32, b: u32) -> Self::Output {
//         self.to_i64().wrapping_mul(a.to_i64()) + b.to_i64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u32) -> Self::Output {
//         self.to_i64().wrapping_add(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u32) -> Self::Output {
//         self.to_i64().wrapping_sub(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u32) -> Self::Output {
//         self.to_i64().wrapping_mul(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u32) -> Self::Output {
//         self.to_i64().wrapping_rem(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u32) -> Self::Output {
//         self.to_i64().max(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u32) -> Self::Output {
//         self.to_i64().min(rhs.to_i64())
//     }
// }
// impl NormalOut<u64> for i8 {
//     type Output = i64;
//     #[inline(always)]
//     fn _pow(self, rhs: u64) -> Self::Output {
//         self.to_i64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u64, max: u64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i64>]();
//             let min = min.[<to_ i64>]();
//             let max = max.[<to_ i64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u64, b: u64) -> Self::Output {
//         self.to_i64().wrapping_mul(a.to_i64()) + b.to_i64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u64) -> Self::Output {
//         self.to_i64().wrapping_add(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u64) -> Self::Output {
//         self.to_i64().wrapping_sub(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u64) -> Self::Output {
//         self.to_i64().wrapping_mul(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u64) -> Self::Output {
//         self.to_i64().wrapping_rem(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u64) -> Self::Output {
//         self.to_i64().max(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u64) -> Self::Output {
//         self.to_i64().min(rhs.to_i64())
//     }
// }
// impl NormalOut<bf16> for i8 {
//     type Output = bf16;
//     #[inline(always)]
//     fn _pow(self, rhs: bf16) -> Self::Output {
//         self.to_bf16().powf(rhs.to_bf16())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bf16, max: bf16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ bf16>]();
//             let min = min.[<to_ bf16>]();
//             let max = max.[<to_ bf16>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bf16, b: bf16) -> Self::Output {
//         self.to_bf16() * a.to_bf16() + b.to_bf16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bf16) -> Self::Output {
//         self.to_bf16() + rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bf16) -> Self::Output {
//         self.to_bf16() - rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bf16) -> Self::Output {
//         self.to_bf16() * rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bf16) -> Self::Output {
//         self.to_bf16() % rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bf16) -> Self::Output {
//         self.to_bf16().max(rhs.to_bf16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bf16) -> Self::Output {
//         self.to_bf16().min(rhs.to_bf16())
//     }
// }
// impl NormalOut<isize> for i8 {
//     type Output = isize;
//     #[inline(always)]
//     fn _pow(self, rhs: isize) -> Self::Output {
//         self.to_isize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: isize, max: isize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ isize>]();
//             let min = min.[<to_ isize>]();
//             let max = max.[<to_ isize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: isize, b: isize) -> Self::Output {
//         self.to_isize().wrapping_mul(a.to_isize()) + b.to_isize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_add(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_sub(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_mul(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_rem(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: isize) -> Self::Output {
//         self.to_isize().max(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: isize) -> Self::Output {
//         self.to_isize().min(rhs.to_isize())
//     }
// }
// impl NormalOut<usize> for i8 {
//     type Output = isize;
//     #[inline(always)]
//     fn _pow(self, rhs: usize) -> Self::Output {
//         self.to_isize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: usize, max: usize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ isize>]();
//             let min = min.[<to_ isize>]();
//             let max = max.[<to_ isize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: usize, b: usize) -> Self::Output {
//         self.to_isize().wrapping_mul(a.to_isize()) + b.to_isize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: usize) -> Self::Output {
//         self.to_isize().wrapping_add(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: usize) -> Self::Output {
//         self.to_isize().wrapping_sub(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: usize) -> Self::Output {
//         self.to_isize().wrapping_mul(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: usize) -> Self::Output {
//         self.to_isize().wrapping_rem(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: usize) -> Self::Output {
//         self.to_isize().max(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: usize) -> Self::Output {
//         self.to_isize().min(rhs.to_isize())
//     }
// }
// impl NormalOut<Complex32> for i8 {
//     type Output = Complex32;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32().powc(rhs.to_complex32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex32, max: Complex32) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex32>]();
//             let min = min.[<to_ complex32>]();
//             let max = max.[<to_ complex32>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex32::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex32, b: Complex32) -> Self::Output {
//         self.to_complex32() * a.to_complex32() + b.to_complex32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() + rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() - rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() * rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() % rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex32) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex32) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<Complex64> for i8 {
//     type Output = Complex64;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64().powc(rhs.to_complex64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex64, max: Complex64) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex64>]();
//             let min = min.[<to_ complex64>]();
//             let max = max.[<to_ complex64>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex64::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex64, b: Complex64) -> Self::Output {
//         self.to_complex64() * a.to_complex64() + b.to_complex64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() + rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() - rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() * rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() % rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex64) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex64) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<bool> for i16 {
//     type Output = i16;
//     #[inline(always)]
//     fn _pow(self, rhs: bool) -> Self::Output {
//         self.to_i16().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bool, max: bool) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i16>]();
//             let min = min.[<to_ i16>]();
//             let max = max.[<to_ i16>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bool, b: bool) -> Self::Output {
//         self.to_i16().wrapping_mul(a.to_i16()) + b.to_i16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bool) -> Self::Output {
//         self.to_i16().wrapping_add(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bool) -> Self::Output {
//         self.to_i16().wrapping_sub(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bool) -> Self::Output {
//         self.to_i16().wrapping_mul(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bool) -> Self::Output {
//         self.to_i16().wrapping_rem(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bool) -> Self::Output {
//         self.to_i16().max(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bool) -> Self::Output {
//         self.to_i16().min(rhs.to_i16())
//     }
// }
// impl NormalOut<half::f16> for i16 {
//     type Output = half::f16;
//     #[inline(always)]
//     fn _pow(self, rhs: half::f16) -> Self::Output {
//         self.to_f16().powf(rhs.to_f16())
//     }
//     #[inline(always)]
//     fn _clip(self, min: half::f16, max: half::f16) -> Self::Output {
//         paste::paste! {
//             let a = self.to_f16();
//             let min = min.to_f16();
//             let max = max.to_f16();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: half::f16, b: half::f16) -> Self::Output {
//         self.to_f16() * a.to_f16() + b.to_f16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: half::f16) -> Self::Output {
//         self.to_f16() + rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: half::f16) -> Self::Output {
//         self.to_f16() - rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: half::f16) -> Self::Output {
//         self.to_f16() * rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: half::f16) -> Self::Output {
//         self.to_f16() % rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: half::f16) -> Self::Output {
//         self.to_f16().max(rhs.to_f16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: half::f16) -> Self::Output {
//         self.to_f16().min(rhs.to_f16())
//     }
// }
// impl NormalOut<f32> for i16 {
//     type Output = f32;
//     #[inline(always)]
//     fn _pow(self, rhs: f32) -> Self::Output {
//         self.to_f32().powf(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f32, max: f32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f32>]();
//             let min = min.[<to_ f32>]();
//             let max = max.[<to_ f32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f32, b: f32) -> Self::Output {
//         self.to_f32() * a.to_f32() + b.to_f32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f32) -> Self::Output {
//         self.to_f32() + rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f32) -> Self::Output {
//         self.to_f32() - rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f32) -> Self::Output {
//         self.to_f32() * rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f32) -> Self::Output {
//         self.to_f32() % rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f32) -> Self::Output {
//         self.to_f32().max(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f32) -> Self::Output {
//         self.to_f32().min(rhs.to_f32())
//     }
// }
// impl NormalOut<f64> for i16 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: f64) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f64, max: f64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f64, b: f64) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f64) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f64) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f64) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f64) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f64) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f64) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<i8> for i16 {
//     type Output = i16;
//     #[inline(always)]
//     fn _pow(self, rhs: i8) -> Self::Output {
//         self.to_i16().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i8, max: i8) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i16>]();
//             let min = min.[<to_ i16>]();
//             let max = max.[<to_ i16>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i8, b: i8) -> Self::Output {
//         self.to_i16().wrapping_mul(a.to_i16()) + b.to_i16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i8) -> Self::Output {
//         self.to_i16().wrapping_add(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i8) -> Self::Output {
//         self.to_i16().wrapping_sub(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i8) -> Self::Output {
//         self.to_i16().wrapping_mul(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i8) -> Self::Output {
//         self.to_i16().wrapping_rem(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i8) -> Self::Output {
//         self.to_i16().max(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i8) -> Self::Output {
//         self.to_i16().min(rhs.to_i16())
//     }
// }
// impl NormalOut<i16> for i16 {
//     type Output = i16;
//     #[inline(always)]
//     fn _pow(self, rhs: i16) -> Self::Output {
//         self.to_i16().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i16, max: i16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i16>]();
//             let min = min.[<to_ i16>]();
//             let max = max.[<to_ i16>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i16, b: i16) -> Self::Output {
//         self.to_i16().wrapping_mul(a.to_i16()) + b.to_i16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i16) -> Self::Output {
//         self.to_i16().wrapping_add(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i16) -> Self::Output {
//         self.to_i16().wrapping_sub(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i16) -> Self::Output {
//         self.to_i16().wrapping_mul(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i16) -> Self::Output {
//         self.to_i16().wrapping_rem(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i16) -> Self::Output {
//         self.to_i16().max(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i16) -> Self::Output {
//         self.to_i16().min(rhs.to_i16())
//     }
// }
// impl NormalOut<i32> for i16 {
//     type Output = i32;
//     #[inline(always)]
//     fn _pow(self, rhs: i32) -> Self::Output {
//         self.to_i32().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i32, max: i32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i32>]();
//             let min = min.[<to_ i32>]();
//             let max = max.[<to_ i32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i32, b: i32) -> Self::Output {
//         self.to_i32().wrapping_mul(a.to_i32()) + b.to_i32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i32) -> Self::Output {
//         self.to_i32().wrapping_add(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i32) -> Self::Output {
//         self.to_i32().wrapping_sub(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i32) -> Self::Output {
//         self.to_i32().wrapping_mul(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i32) -> Self::Output {
//         self.to_i32().wrapping_rem(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i32) -> Self::Output {
//         self.to_i32().max(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i32) -> Self::Output {
//         self.to_i32().min(rhs.to_i32())
//     }
// }
// impl NormalOut<i64> for i16 {
//     type Output = i64;
//     #[inline(always)]
//     fn _pow(self, rhs: i64) -> Self::Output {
//         self.to_i64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i64, max: i64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i64>]();
//             let min = min.[<to_ i64>]();
//             let max = max.[<to_ i64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i64, b: i64) -> Self::Output {
//         self.to_i64().wrapping_mul(a.to_i64()) + b.to_i64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_add(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_sub(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_mul(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_rem(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i64) -> Self::Output {
//         self.to_i64().max(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i64) -> Self::Output {
//         self.to_i64().min(rhs.to_i64())
//     }
// }
// impl NormalOut<u8> for i16 {
//     type Output = i16;
//     #[inline(always)]
//     fn _pow(self, rhs: u8) -> Self::Output {
//         self.to_i16().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u8, max: u8) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i16>]();
//             let min = min.[<to_ i16>]();
//             let max = max.[<to_ i16>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u8, b: u8) -> Self::Output {
//         self.to_i16().wrapping_mul(a.to_i16()) + b.to_i16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u8) -> Self::Output {
//         self.to_i16().wrapping_add(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u8) -> Self::Output {
//         self.to_i16().wrapping_sub(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u8) -> Self::Output {
//         self.to_i16().wrapping_mul(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u8) -> Self::Output {
//         self.to_i16().wrapping_rem(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u8) -> Self::Output {
//         self.to_i16().max(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u8) -> Self::Output {
//         self.to_i16().min(rhs.to_i16())
//     }
// }
// impl NormalOut<u16> for i16 {
//     type Output = i32;
//     #[inline(always)]
//     fn _pow(self, rhs: u16) -> Self::Output {
//         self.to_i32().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u16, max: u16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i32>]();
//             let min = min.[<to_ i32>]();
//             let max = max.[<to_ i32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u16, b: u16) -> Self::Output {
//         self.to_i32().wrapping_mul(a.to_i32()) + b.to_i32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u16) -> Self::Output {
//         self.to_i32().wrapping_add(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u16) -> Self::Output {
//         self.to_i32().wrapping_sub(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u16) -> Self::Output {
//         self.to_i32().wrapping_mul(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u16) -> Self::Output {
//         self.to_i32().wrapping_rem(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u16) -> Self::Output {
//         self.to_i32().max(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u16) -> Self::Output {
//         self.to_i32().min(rhs.to_i32())
//     }
// }
// impl NormalOut<u32> for i16 {
//     type Output = i64;
//     #[inline(always)]
//     fn _pow(self, rhs: u32) -> Self::Output {
//         self.to_i64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u32, max: u32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i64>]();
//             let min = min.[<to_ i64>]();
//             let max = max.[<to_ i64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u32, b: u32) -> Self::Output {
//         self.to_i64().wrapping_mul(a.to_i64()) + b.to_i64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u32) -> Self::Output {
//         self.to_i64().wrapping_add(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u32) -> Self::Output {
//         self.to_i64().wrapping_sub(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u32) -> Self::Output {
//         self.to_i64().wrapping_mul(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u32) -> Self::Output {
//         self.to_i64().wrapping_rem(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u32) -> Self::Output {
//         self.to_i64().max(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u32) -> Self::Output {
//         self.to_i64().min(rhs.to_i64())
//     }
// }
// impl NormalOut<u64> for i16 {
//     type Output = i64;
//     #[inline(always)]
//     fn _pow(self, rhs: u64) -> Self::Output {
//         self.to_i64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u64, max: u64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i64>]();
//             let min = min.[<to_ i64>]();
//             let max = max.[<to_ i64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u64, b: u64) -> Self::Output {
//         self.to_i64().wrapping_mul(a.to_i64()) + b.to_i64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u64) -> Self::Output {
//         self.to_i64().wrapping_add(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u64) -> Self::Output {
//         self.to_i64().wrapping_sub(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u64) -> Self::Output {
//         self.to_i64().wrapping_mul(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u64) -> Self::Output {
//         self.to_i64().wrapping_rem(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u64) -> Self::Output {
//         self.to_i64().max(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u64) -> Self::Output {
//         self.to_i64().min(rhs.to_i64())
//     }
// }
// impl NormalOut<bf16> for i16 {
//     type Output = bf16;
//     #[inline(always)]
//     fn _pow(self, rhs: bf16) -> Self::Output {
//         self.to_bf16().powf(rhs.to_bf16())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bf16, max: bf16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ bf16>]();
//             let min = min.[<to_ bf16>]();
//             let max = max.[<to_ bf16>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bf16, b: bf16) -> Self::Output {
//         self.to_bf16() * a.to_bf16() + b.to_bf16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bf16) -> Self::Output {
//         self.to_bf16() + rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bf16) -> Self::Output {
//         self.to_bf16() - rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bf16) -> Self::Output {
//         self.to_bf16() * rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bf16) -> Self::Output {
//         self.to_bf16() % rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bf16) -> Self::Output {
//         self.to_bf16().max(rhs.to_bf16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bf16) -> Self::Output {
//         self.to_bf16().min(rhs.to_bf16())
//     }
// }
// impl NormalOut<isize> for i16 {
//     type Output = isize;
//     #[inline(always)]
//     fn _pow(self, rhs: isize) -> Self::Output {
//         self.to_isize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: isize, max: isize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ isize>]();
//             let min = min.[<to_ isize>]();
//             let max = max.[<to_ isize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: isize, b: isize) -> Self::Output {
//         self.to_isize().wrapping_mul(a.to_isize()) + b.to_isize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_add(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_sub(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_mul(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_rem(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: isize) -> Self::Output {
//         self.to_isize().max(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: isize) -> Self::Output {
//         self.to_isize().min(rhs.to_isize())
//     }
// }
// impl NormalOut<usize> for i16 {
//     type Output = isize;
//     #[inline(always)]
//     fn _pow(self, rhs: usize) -> Self::Output {
//         self.to_isize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: usize, max: usize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ isize>]();
//             let min = min.[<to_ isize>]();
//             let max = max.[<to_ isize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: usize, b: usize) -> Self::Output {
//         self.to_isize().wrapping_mul(a.to_isize()) + b.to_isize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: usize) -> Self::Output {
//         self.to_isize().wrapping_add(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: usize) -> Self::Output {
//         self.to_isize().wrapping_sub(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: usize) -> Self::Output {
//         self.to_isize().wrapping_mul(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: usize) -> Self::Output {
//         self.to_isize().wrapping_rem(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: usize) -> Self::Output {
//         self.to_isize().max(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: usize) -> Self::Output {
//         self.to_isize().min(rhs.to_isize())
//     }
// }
// impl NormalOut<Complex32> for i16 {
//     type Output = Complex32;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32().powc(rhs.to_complex32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex32, max: Complex32) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex32>]();
//             let min = min.[<to_ complex32>]();
//             let max = max.[<to_ complex32>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex32::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex32, b: Complex32) -> Self::Output {
//         self.to_complex32() * a.to_complex32() + b.to_complex32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() + rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() - rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() * rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() % rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex32) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex32) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<Complex64> for i16 {
//     type Output = Complex64;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64().powc(rhs.to_complex64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex64, max: Complex64) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex64>]();
//             let min = min.[<to_ complex64>]();
//             let max = max.[<to_ complex64>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex64::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex64, b: Complex64) -> Self::Output {
//         self.to_complex64() * a.to_complex64() + b.to_complex64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() + rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() - rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() * rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() % rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex64) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex64) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<bool> for i32 {
//     type Output = i32;
//     #[inline(always)]
//     fn _pow(self, rhs: bool) -> Self::Output {
//         self.to_i32().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bool, max: bool) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i32>]();
//             let min = min.[<to_ i32>]();
//             let max = max.[<to_ i32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bool, b: bool) -> Self::Output {
//         self.to_i32().wrapping_mul(a.to_i32()) + b.to_i32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bool) -> Self::Output {
//         self.to_i32().wrapping_add(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bool) -> Self::Output {
//         self.to_i32().wrapping_sub(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bool) -> Self::Output {
//         self.to_i32().wrapping_mul(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bool) -> Self::Output {
//         self.to_i32().wrapping_rem(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bool) -> Self::Output {
//         self.to_i32().max(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bool) -> Self::Output {
//         self.to_i32().min(rhs.to_i32())
//     }
// }
// impl NormalOut<half::f16> for i32 {
//     type Output = f32;
//     #[inline(always)]
//     fn _pow(self, rhs: half::f16) -> Self::Output {
//         self.to_f32().powf(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: half::f16, max: half::f16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f32>]();
//             let min = min.[<to_ f32>]();
//             let max = max.[<to_ f32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: half::f16, b: half::f16) -> Self::Output {
//         self.to_f32() * a.to_f32() + b.to_f32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: half::f16) -> Self::Output {
//         self.to_f32() + rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: half::f16) -> Self::Output {
//         self.to_f32() - rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: half::f16) -> Self::Output {
//         self.to_f32() * rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: half::f16) -> Self::Output {
//         self.to_f32() % rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: half::f16) -> Self::Output {
//         self.to_f32().max(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: half::f16) -> Self::Output {
//         self.to_f32().min(rhs.to_f32())
//     }
// }
// impl NormalOut<f32> for i32 {
//     type Output = f32;
//     #[inline(always)]
//     fn _pow(self, rhs: f32) -> Self::Output {
//         self.to_f32().powf(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f32, max: f32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f32>]();
//             let min = min.[<to_ f32>]();
//             let max = max.[<to_ f32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f32, b: f32) -> Self::Output {
//         self.to_f32() * a.to_f32() + b.to_f32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f32) -> Self::Output {
//         self.to_f32() + rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f32) -> Self::Output {
//         self.to_f32() - rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f32) -> Self::Output {
//         self.to_f32() * rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f32) -> Self::Output {
//         self.to_f32() % rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f32) -> Self::Output {
//         self.to_f32().max(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f32) -> Self::Output {
//         self.to_f32().min(rhs.to_f32())
//     }
// }
// impl NormalOut<f64> for i32 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: f64) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f64, max: f64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f64, b: f64) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f64) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f64) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f64) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f64) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f64) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f64) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<i8> for i32 {
//     type Output = i32;
//     #[inline(always)]
//     fn _pow(self, rhs: i8) -> Self::Output {
//         self.to_i32().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i8, max: i8) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i32>]();
//             let min = min.[<to_ i32>]();
//             let max = max.[<to_ i32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i8, b: i8) -> Self::Output {
//         self.to_i32().wrapping_mul(a.to_i32()) + b.to_i32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i8) -> Self::Output {
//         self.to_i32().wrapping_add(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i8) -> Self::Output {
//         self.to_i32().wrapping_sub(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i8) -> Self::Output {
//         self.to_i32().wrapping_mul(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i8) -> Self::Output {
//         self.to_i32().wrapping_rem(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i8) -> Self::Output {
//         self.to_i32().max(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i8) -> Self::Output {
//         self.to_i32().min(rhs.to_i32())
//     }
// }
// impl NormalOut<i16> for i32 {
//     type Output = i32;
//     #[inline(always)]
//     fn _pow(self, rhs: i16) -> Self::Output {
//         self.to_i32().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i16, max: i16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i32>]();
//             let min = min.[<to_ i32>]();
//             let max = max.[<to_ i32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i16, b: i16) -> Self::Output {
//         self.to_i32().wrapping_mul(a.to_i32()) + b.to_i32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i16) -> Self::Output {
//         self.to_i32().wrapping_add(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i16) -> Self::Output {
//         self.to_i32().wrapping_sub(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i16) -> Self::Output {
//         self.to_i32().wrapping_mul(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i16) -> Self::Output {
//         self.to_i32().wrapping_rem(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i16) -> Self::Output {
//         self.to_i32().max(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i16) -> Self::Output {
//         self.to_i32().min(rhs.to_i32())
//     }
// }
// impl NormalOut<i32> for i32 {
//     type Output = i32;
//     #[inline(always)]
//     fn _pow(self, rhs: i32) -> Self::Output {
//         self.to_i32().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i32, max: i32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i32>]();
//             let min = min.[<to_ i32>]();
//             let max = max.[<to_ i32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i32, b: i32) -> Self::Output {
//         self.to_i32().wrapping_mul(a.to_i32()) + b.to_i32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i32) -> Self::Output {
//         self.to_i32().wrapping_add(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i32) -> Self::Output {
//         self.to_i32().wrapping_sub(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i32) -> Self::Output {
//         self.to_i32().wrapping_mul(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i32) -> Self::Output {
//         self.to_i32().wrapping_rem(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i32) -> Self::Output {
//         self.to_i32().max(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i32) -> Self::Output {
//         self.to_i32().min(rhs.to_i32())
//     }
// }
// impl NormalOut<i64> for i32 {
//     type Output = i64;
//     #[inline(always)]
//     fn _pow(self, rhs: i64) -> Self::Output {
//         self.to_i64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i64, max: i64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i64>]();
//             let min = min.[<to_ i64>]();
//             let max = max.[<to_ i64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i64, b: i64) -> Self::Output {
//         self.to_i64().wrapping_mul(a.to_i64()) + b.to_i64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_add(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_sub(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_mul(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_rem(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i64) -> Self::Output {
//         self.to_i64().max(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i64) -> Self::Output {
//         self.to_i64().min(rhs.to_i64())
//     }
// }
// impl NormalOut<u8> for i32 {
//     type Output = i32;
//     #[inline(always)]
//     fn _pow(self, rhs: u8) -> Self::Output {
//         self.to_i32().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u8, max: u8) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i32>]();
//             let min = min.[<to_ i32>]();
//             let max = max.[<to_ i32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u8, b: u8) -> Self::Output {
//         self.to_i32().wrapping_mul(a.to_i32()) + b.to_i32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u8) -> Self::Output {
//         self.to_i32().wrapping_add(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u8) -> Self::Output {
//         self.to_i32().wrapping_sub(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u8) -> Self::Output {
//         self.to_i32().wrapping_mul(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u8) -> Self::Output {
//         self.to_i32().wrapping_rem(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u8) -> Self::Output {
//         self.to_i32().max(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u8) -> Self::Output {
//         self.to_i32().min(rhs.to_i32())
//     }
// }
// impl NormalOut<u16> for i32 {
//     type Output = i32;
//     #[inline(always)]
//     fn _pow(self, rhs: u16) -> Self::Output {
//         self.to_i32().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u16, max: u16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i32>]();
//             let min = min.[<to_ i32>]();
//             let max = max.[<to_ i32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u16, b: u16) -> Self::Output {
//         self.to_i32().wrapping_mul(a.to_i32()) + b.to_i32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u16) -> Self::Output {
//         self.to_i32().wrapping_add(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u16) -> Self::Output {
//         self.to_i32().wrapping_sub(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u16) -> Self::Output {
//         self.to_i32().wrapping_mul(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u16) -> Self::Output {
//         self.to_i32().wrapping_rem(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u16) -> Self::Output {
//         self.to_i32().max(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u16) -> Self::Output {
//         self.to_i32().min(rhs.to_i32())
//     }
// }
// impl NormalOut<u32> for i32 {
//     type Output = i64;
//     #[inline(always)]
//     fn _pow(self, rhs: u32) -> Self::Output {
//         self.to_i64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u32, max: u32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i64>]();
//             let min = min.[<to_ i64>]();
//             let max = max.[<to_ i64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u32, b: u32) -> Self::Output {
//         self.to_i64().wrapping_mul(a.to_i64()) + b.to_i64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u32) -> Self::Output {
//         self.to_i64().wrapping_add(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u32) -> Self::Output {
//         self.to_i64().wrapping_sub(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u32) -> Self::Output {
//         self.to_i64().wrapping_mul(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u32) -> Self::Output {
//         self.to_i64().wrapping_rem(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u32) -> Self::Output {
//         self.to_i64().max(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u32) -> Self::Output {
//         self.to_i64().min(rhs.to_i64())
//     }
// }
// impl NormalOut<u64> for i32 {
//     type Output = i64;
//     #[inline(always)]
//     fn _pow(self, rhs: u64) -> Self::Output {
//         self.to_i64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u64, max: u64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i64>]();
//             let min = min.[<to_ i64>]();
//             let max = max.[<to_ i64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u64, b: u64) -> Self::Output {
//         self.to_i64().wrapping_mul(a.to_i64()) + b.to_i64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u64) -> Self::Output {
//         self.to_i64().wrapping_add(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u64) -> Self::Output {
//         self.to_i64().wrapping_sub(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u64) -> Self::Output {
//         self.to_i64().wrapping_mul(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u64) -> Self::Output {
//         self.to_i64().wrapping_rem(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u64) -> Self::Output {
//         self.to_i64().max(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u64) -> Self::Output {
//         self.to_i64().min(rhs.to_i64())
//     }
// }
// impl NormalOut<bf16> for i32 {
//     type Output = f32;
//     #[inline(always)]
//     fn _pow(self, rhs: bf16) -> Self::Output {
//         self.to_f32().powf(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bf16, max: bf16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f32>]();
//             let min = min.[<to_ f32>]();
//             let max = max.[<to_ f32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bf16, b: bf16) -> Self::Output {
//         self.to_f32() * a.to_f32() + b.to_f32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bf16) -> Self::Output {
//         self.to_f32() + rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bf16) -> Self::Output {
//         self.to_f32() - rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bf16) -> Self::Output {
//         self.to_f32() * rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bf16) -> Self::Output {
//         self.to_f32() % rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bf16) -> Self::Output {
//         self.to_f32().max(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bf16) -> Self::Output {
//         self.to_f32().min(rhs.to_f32())
//     }
// }
// impl NormalOut<isize> for i32 {
//     type Output = isize;
//     #[inline(always)]
//     fn _pow(self, rhs: isize) -> Self::Output {
//         self.to_isize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: isize, max: isize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ isize>]();
//             let min = min.[<to_ isize>]();
//             let max = max.[<to_ isize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: isize, b: isize) -> Self::Output {
//         self.to_isize().wrapping_mul(a.to_isize()) + b.to_isize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_add(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_sub(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_mul(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_rem(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: isize) -> Self::Output {
//         self.to_isize().max(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: isize) -> Self::Output {
//         self.to_isize().min(rhs.to_isize())
//     }
// }
// impl NormalOut<usize> for i32 {
//     type Output = isize;
//     #[inline(always)]
//     fn _pow(self, rhs: usize) -> Self::Output {
//         self.to_isize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: usize, max: usize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ isize>]();
//             let min = min.[<to_ isize>]();
//             let max = max.[<to_ isize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: usize, b: usize) -> Self::Output {
//         self.to_isize().wrapping_mul(a.to_isize()) + b.to_isize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: usize) -> Self::Output {
//         self.to_isize().wrapping_add(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: usize) -> Self::Output {
//         self.to_isize().wrapping_sub(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: usize) -> Self::Output {
//         self.to_isize().wrapping_mul(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: usize) -> Self::Output {
//         self.to_isize().wrapping_rem(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: usize) -> Self::Output {
//         self.to_isize().max(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: usize) -> Self::Output {
//         self.to_isize().min(rhs.to_isize())
//     }
// }
// impl NormalOut<Complex32> for i32 {
//     type Output = Complex32;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32().powc(rhs.to_complex32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex32, max: Complex32) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex32>]();
//             let min = min.[<to_ complex32>]();
//             let max = max.[<to_ complex32>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex32::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex32, b: Complex32) -> Self::Output {
//         self.to_complex32() * a.to_complex32() + b.to_complex32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() + rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() - rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() * rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() % rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex32) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex32) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<Complex64> for i32 {
//     type Output = Complex64;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64().powc(rhs.to_complex64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex64, max: Complex64) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex64>]();
//             let min = min.[<to_ complex64>]();
//             let max = max.[<to_ complex64>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex64::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex64, b: Complex64) -> Self::Output {
//         self.to_complex64() * a.to_complex64() + b.to_complex64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() + rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() - rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() * rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() % rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex64) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex64) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<bool> for i64 {
//     type Output = i64;
//     #[inline(always)]
//     fn _pow(self, rhs: bool) -> Self::Output {
//         self.to_i64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bool, max: bool) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i64>]();
//             let min = min.[<to_ i64>]();
//             let max = max.[<to_ i64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bool, b: bool) -> Self::Output {
//         self.to_i64().wrapping_mul(a.to_i64()) + b.to_i64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bool) -> Self::Output {
//         self.to_i64().wrapping_add(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bool) -> Self::Output {
//         self.to_i64().wrapping_sub(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bool) -> Self::Output {
//         self.to_i64().wrapping_mul(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bool) -> Self::Output {
//         self.to_i64().wrapping_rem(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bool) -> Self::Output {
//         self.to_i64().max(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bool) -> Self::Output {
//         self.to_i64().min(rhs.to_i64())
//     }
// }
// impl NormalOut<half::f16> for i64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: half::f16) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: half::f16, max: half::f16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: half::f16, b: half::f16) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: half::f16) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: half::f16) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: half::f16) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: half::f16) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: half::f16) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: half::f16) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<f32> for i64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: f32) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f32, max: f32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f32, b: f32) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f32) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f32) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f32) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f32) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f32) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f32) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<f64> for i64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: f64) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f64, max: f64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f64, b: f64) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f64) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f64) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f64) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f64) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f64) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f64) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<i8> for i64 {
//     type Output = i64;
//     #[inline(always)]
//     fn _pow(self, rhs: i8) -> Self::Output {
//         self.to_i64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i8, max: i8) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i64>]();
//             let min = min.[<to_ i64>]();
//             let max = max.[<to_ i64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i8, b: i8) -> Self::Output {
//         self.to_i64().wrapping_mul(a.to_i64()) + b.to_i64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i8) -> Self::Output {
//         self.to_i64().wrapping_add(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i8) -> Self::Output {
//         self.to_i64().wrapping_sub(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i8) -> Self::Output {
//         self.to_i64().wrapping_mul(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i8) -> Self::Output {
//         self.to_i64().wrapping_rem(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i8) -> Self::Output {
//         self.to_i64().max(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i8) -> Self::Output {
//         self.to_i64().min(rhs.to_i64())
//     }
// }
// impl NormalOut<i16> for i64 {
//     type Output = i64;
//     #[inline(always)]
//     fn _pow(self, rhs: i16) -> Self::Output {
//         self.to_i64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i16, max: i16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i64>]();
//             let min = min.[<to_ i64>]();
//             let max = max.[<to_ i64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i16, b: i16) -> Self::Output {
//         self.to_i64().wrapping_mul(a.to_i64()) + b.to_i64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i16) -> Self::Output {
//         self.to_i64().wrapping_add(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i16) -> Self::Output {
//         self.to_i64().wrapping_sub(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i16) -> Self::Output {
//         self.to_i64().wrapping_mul(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i16) -> Self::Output {
//         self.to_i64().wrapping_rem(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i16) -> Self::Output {
//         self.to_i64().max(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i16) -> Self::Output {
//         self.to_i64().min(rhs.to_i64())
//     }
// }
// impl NormalOut<i32> for i64 {
//     type Output = i64;
//     #[inline(always)]
//     fn _pow(self, rhs: i32) -> Self::Output {
//         self.to_i64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i32, max: i32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i64>]();
//             let min = min.[<to_ i64>]();
//             let max = max.[<to_ i64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i32, b: i32) -> Self::Output {
//         self.to_i64().wrapping_mul(a.to_i64()) + b.to_i64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i32) -> Self::Output {
//         self.to_i64().wrapping_add(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i32) -> Self::Output {
//         self.to_i64().wrapping_sub(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i32) -> Self::Output {
//         self.to_i64().wrapping_mul(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i32) -> Self::Output {
//         self.to_i64().wrapping_rem(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i32) -> Self::Output {
//         self.to_i64().max(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i32) -> Self::Output {
//         self.to_i64().min(rhs.to_i64())
//     }
// }
// impl NormalOut<i64> for i64 {
//     type Output = i64;
//     #[inline(always)]
//     fn _pow(self, rhs: i64) -> Self::Output {
//         self.to_i64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i64, max: i64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i64>]();
//             let min = min.[<to_ i64>]();
//             let max = max.[<to_ i64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i64, b: i64) -> Self::Output {
//         self.to_i64().wrapping_mul(a.to_i64()) + b.to_i64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_add(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_sub(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_mul(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_rem(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i64) -> Self::Output {
//         self.to_i64().max(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i64) -> Self::Output {
//         self.to_i64().min(rhs.to_i64())
//     }
// }
// impl NormalOut<u8> for i64 {
//     type Output = i64;
//     #[inline(always)]
//     fn _pow(self, rhs: u8) -> Self::Output {
//         self.to_i64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u8, max: u8) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i64>]();
//             let min = min.[<to_ i64>]();
//             let max = max.[<to_ i64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u8, b: u8) -> Self::Output {
//         self.to_i64().wrapping_mul(a.to_i64()) + b.to_i64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u8) -> Self::Output {
//         self.to_i64().wrapping_add(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u8) -> Self::Output {
//         self.to_i64().wrapping_sub(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u8) -> Self::Output {
//         self.to_i64().wrapping_mul(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u8) -> Self::Output {
//         self.to_i64().wrapping_rem(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u8) -> Self::Output {
//         self.to_i64().max(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u8) -> Self::Output {
//         self.to_i64().min(rhs.to_i64())
//     }
// }
// impl NormalOut<u16> for i64 {
//     type Output = i64;
//     #[inline(always)]
//     fn _pow(self, rhs: u16) -> Self::Output {
//         self.to_i64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u16, max: u16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i64>]();
//             let min = min.[<to_ i64>]();
//             let max = max.[<to_ i64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u16, b: u16) -> Self::Output {
//         self.to_i64().wrapping_mul(a.to_i64()) + b.to_i64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u16) -> Self::Output {
//         self.to_i64().wrapping_add(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u16) -> Self::Output {
//         self.to_i64().wrapping_sub(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u16) -> Self::Output {
//         self.to_i64().wrapping_mul(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u16) -> Self::Output {
//         self.to_i64().wrapping_rem(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u16) -> Self::Output {
//         self.to_i64().max(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u16) -> Self::Output {
//         self.to_i64().min(rhs.to_i64())
//     }
// }
// impl NormalOut<u32> for i64 {
//     type Output = i64;
//     #[inline(always)]
//     fn _pow(self, rhs: u32) -> Self::Output {
//         self.to_i64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u32, max: u32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i64>]();
//             let min = min.[<to_ i64>]();
//             let max = max.[<to_ i64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u32, b: u32) -> Self::Output {
//         self.to_i64().wrapping_mul(a.to_i64()) + b.to_i64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u32) -> Self::Output {
//         self.to_i64().wrapping_add(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u32) -> Self::Output {
//         self.to_i64().wrapping_sub(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u32) -> Self::Output {
//         self.to_i64().wrapping_mul(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u32) -> Self::Output {
//         self.to_i64().wrapping_rem(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u32) -> Self::Output {
//         self.to_i64().max(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u32) -> Self::Output {
//         self.to_i64().min(rhs.to_i64())
//     }
// }
// impl NormalOut<u64> for i64 {
//     type Output = i64;
//     #[inline(always)]
//     fn _pow(self, rhs: u64) -> Self::Output {
//         self.to_i64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u64, max: u64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i64>]();
//             let min = min.[<to_ i64>]();
//             let max = max.[<to_ i64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u64, b: u64) -> Self::Output {
//         self.to_i64().wrapping_mul(a.to_i64()) + b.to_i64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u64) -> Self::Output {
//         self.to_i64().wrapping_add(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u64) -> Self::Output {
//         self.to_i64().wrapping_sub(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u64) -> Self::Output {
//         self.to_i64().wrapping_mul(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u64) -> Self::Output {
//         self.to_i64().wrapping_rem(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u64) -> Self::Output {
//         self.to_i64().max(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u64) -> Self::Output {
//         self.to_i64().min(rhs.to_i64())
//     }
// }
// impl NormalOut<bf16> for i64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: bf16) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bf16, max: bf16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bf16, b: bf16) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bf16) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bf16) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bf16) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bf16) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bf16) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bf16) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<isize> for i64 {
//     type Output = isize;
//     #[inline(always)]
//     fn _pow(self, rhs: isize) -> Self::Output {
//         self.to_isize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: isize, max: isize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ isize>]();
//             let min = min.[<to_ isize>]();
//             let max = max.[<to_ isize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: isize, b: isize) -> Self::Output {
//         self.to_isize().wrapping_mul(a.to_isize()) + b.to_isize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_add(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_sub(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_mul(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_rem(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: isize) -> Self::Output {
//         self.to_isize().max(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: isize) -> Self::Output {
//         self.to_isize().min(rhs.to_isize())
//     }
// }
// impl NormalOut<usize> for i64 {
//     type Output = isize;
//     #[inline(always)]
//     fn _pow(self, rhs: usize) -> Self::Output {
//         self.to_isize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: usize, max: usize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ isize>]();
//             let min = min.[<to_ isize>]();
//             let max = max.[<to_ isize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: usize, b: usize) -> Self::Output {
//         self.to_isize().wrapping_mul(a.to_isize()) + b.to_isize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: usize) -> Self::Output {
//         self.to_isize().wrapping_add(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: usize) -> Self::Output {
//         self.to_isize().wrapping_sub(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: usize) -> Self::Output {
//         self.to_isize().wrapping_mul(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: usize) -> Self::Output {
//         self.to_isize().wrapping_rem(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: usize) -> Self::Output {
//         self.to_isize().max(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: usize) -> Self::Output {
//         self.to_isize().min(rhs.to_isize())
//     }
// }
// impl NormalOut<Complex32> for i64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex32) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex32, max: Complex32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex32, b: Complex32) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex32) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex32) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex32) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex32) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex32) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex32) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<Complex64> for i64 {
//     type Output = Complex64;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64().powc(rhs.to_complex64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex64, max: Complex64) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex64>]();
//             let min = min.[<to_ complex64>]();
//             let max = max.[<to_ complex64>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex64::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex64, b: Complex64) -> Self::Output {
//         self.to_complex64() * a.to_complex64() + b.to_complex64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() + rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() - rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() * rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() % rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex64) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex64) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<bool> for u8 {
//     type Output = u8;
//     #[inline(always)]
//     fn _pow(self, rhs: bool) -> Self::Output {
//         self.to_u8().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bool, max: bool) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ u8>]();
//             let min = min.[<to_ u8>]();
//             let max = max.[<to_ u8>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bool, b: bool) -> Self::Output {
//         self.to_u8().wrapping_mul(a.to_u8()) + b.to_u8()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bool) -> Self::Output {
//         self.to_u8().wrapping_add(rhs.to_u8())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bool) -> Self::Output {
//         self.to_u8().wrapping_sub(rhs.to_u8())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bool) -> Self::Output {
//         self.to_u8().wrapping_mul(rhs.to_u8())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bool) -> Self::Output {
//         self.to_u8().wrapping_rem(rhs.to_u8())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bool) -> Self::Output {
//         self.to_u8().max(rhs.to_u8())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bool) -> Self::Output {
//         self.to_u8().min(rhs.to_u8())
//     }
// }
// impl NormalOut<half::f16> for u8 {
//     type Output = half::f16;
//     #[inline(always)]
//     fn _pow(self, rhs: half::f16) -> Self::Output {
//         self.to_f16().powf(rhs.to_f16())
//     }
//     #[inline(always)]
//     fn _clip(self, min: half::f16, max: half::f16) -> Self::Output {
//         paste::paste! {
//             let a = self.to_f16();
//             let min = min.to_f16();
//             let max = max.to_f16();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: half::f16, b: half::f16) -> Self::Output {
//         self.to_f16() * a.to_f16() + b.to_f16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: half::f16) -> Self::Output {
//         self.to_f16() + rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: half::f16) -> Self::Output {
//         self.to_f16() - rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: half::f16) -> Self::Output {
//         self.to_f16() * rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: half::f16) -> Self::Output {
//         self.to_f16() % rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: half::f16) -> Self::Output {
//         self.to_f16().max(rhs.to_f16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: half::f16) -> Self::Output {
//         self.to_f16().min(rhs.to_f16())
//     }
// }
// impl NormalOut<f32> for u8 {
//     type Output = f32;
//     #[inline(always)]
//     fn _pow(self, rhs: f32) -> Self::Output {
//         self.to_f32().powf(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f32, max: f32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f32>]();
//             let min = min.[<to_ f32>]();
//             let max = max.[<to_ f32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f32, b: f32) -> Self::Output {
//         self.to_f32() * a.to_f32() + b.to_f32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f32) -> Self::Output {
//         self.to_f32() + rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f32) -> Self::Output {
//         self.to_f32() - rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f32) -> Self::Output {
//         self.to_f32() * rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f32) -> Self::Output {
//         self.to_f32() % rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f32) -> Self::Output {
//         self.to_f32().max(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f32) -> Self::Output {
//         self.to_f32().min(rhs.to_f32())
//     }
// }
// impl NormalOut<f64> for u8 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: f64) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f64, max: f64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f64, b: f64) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f64) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f64) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f64) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f64) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f64) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f64) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<i8> for u8 {
//     type Output = u8;
//     #[inline(always)]
//     fn _pow(self, rhs: i8) -> Self::Output {
//         self.to_u8().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i8, max: i8) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ u8>]();
//             let min = min.[<to_ u8>]();
//             let max = max.[<to_ u8>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i8, b: i8) -> Self::Output {
//         self.to_u8().wrapping_mul(a.to_u8()) + b.to_u8()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i8) -> Self::Output {
//         self.to_u8().wrapping_add(rhs.to_u8())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i8) -> Self::Output {
//         self.to_u8().wrapping_sub(rhs.to_u8())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i8) -> Self::Output {
//         self.to_u8().wrapping_mul(rhs.to_u8())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i8) -> Self::Output {
//         self.to_u8().wrapping_rem(rhs.to_u8())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i8) -> Self::Output {
//         self.to_u8().max(rhs.to_u8())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i8) -> Self::Output {
//         self.to_u8().min(rhs.to_u8())
//     }
// }
// impl NormalOut<i16> for u8 {
//     type Output = i16;
//     #[inline(always)]
//     fn _pow(self, rhs: i16) -> Self::Output {
//         self.to_i16().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i16, max: i16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i16>]();
//             let min = min.[<to_ i16>]();
//             let max = max.[<to_ i16>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i16, b: i16) -> Self::Output {
//         self.to_i16().wrapping_mul(a.to_i16()) + b.to_i16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i16) -> Self::Output {
//         self.to_i16().wrapping_add(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i16) -> Self::Output {
//         self.to_i16().wrapping_sub(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i16) -> Self::Output {
//         self.to_i16().wrapping_mul(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i16) -> Self::Output {
//         self.to_i16().wrapping_rem(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i16) -> Self::Output {
//         self.to_i16().max(rhs.to_i16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i16) -> Self::Output {
//         self.to_i16().min(rhs.to_i16())
//     }
// }
// impl NormalOut<i32> for u8 {
//     type Output = i32;
//     #[inline(always)]
//     fn _pow(self, rhs: i32) -> Self::Output {
//         self.to_i32().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i32, max: i32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i32>]();
//             let min = min.[<to_ i32>]();
//             let max = max.[<to_ i32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i32, b: i32) -> Self::Output {
//         self.to_i32().wrapping_mul(a.to_i32()) + b.to_i32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i32) -> Self::Output {
//         self.to_i32().wrapping_add(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i32) -> Self::Output {
//         self.to_i32().wrapping_sub(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i32) -> Self::Output {
//         self.to_i32().wrapping_mul(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i32) -> Self::Output {
//         self.to_i32().wrapping_rem(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i32) -> Self::Output {
//         self.to_i32().max(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i32) -> Self::Output {
//         self.to_i32().min(rhs.to_i32())
//     }
// }
// impl NormalOut<i64> for u8 {
//     type Output = i64;
//     #[inline(always)]
//     fn _pow(self, rhs: i64) -> Self::Output {
//         self.to_i64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i64, max: i64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i64>]();
//             let min = min.[<to_ i64>]();
//             let max = max.[<to_ i64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i64, b: i64) -> Self::Output {
//         self.to_i64().wrapping_mul(a.to_i64()) + b.to_i64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_add(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_sub(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_mul(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_rem(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i64) -> Self::Output {
//         self.to_i64().max(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i64) -> Self::Output {
//         self.to_i64().min(rhs.to_i64())
//     }
// }
// impl NormalOut<u8> for u8 {
//     type Output = u8;
//     #[inline(always)]
//     fn _pow(self, rhs: u8) -> Self::Output {
//         self.to_u8().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u8, max: u8) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ u8>]();
//             let min = min.[<to_ u8>]();
//             let max = max.[<to_ u8>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u8, b: u8) -> Self::Output {
//         self.to_u8().wrapping_mul(a.to_u8()) + b.to_u8()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u8) -> Self::Output {
//         self.to_u8().wrapping_add(rhs.to_u8())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u8) -> Self::Output {
//         self.to_u8().wrapping_sub(rhs.to_u8())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u8) -> Self::Output {
//         self.to_u8().wrapping_mul(rhs.to_u8())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u8) -> Self::Output {
//         self.to_u8().wrapping_rem(rhs.to_u8())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u8) -> Self::Output {
//         self.to_u8().max(rhs.to_u8())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u8) -> Self::Output {
//         self.to_u8().min(rhs.to_u8())
//     }
// }
// impl NormalOut<u16> for u8 {
//     type Output = u16;
//     #[inline(always)]
//     fn _pow(self, rhs: u16) -> Self::Output {
//         self.to_u16().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u16, max: u16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ u16>]();
//             let min = min.[<to_ u16>]();
//             let max = max.[<to_ u16>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u16, b: u16) -> Self::Output {
//         self.to_u16().wrapping_mul(a.to_u16()) + b.to_u16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u16) -> Self::Output {
//         self.to_u16().wrapping_add(rhs.to_u16())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u16) -> Self::Output {
//         self.to_u16().wrapping_sub(rhs.to_u16())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u16) -> Self::Output {
//         self.to_u16().wrapping_mul(rhs.to_u16())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u16) -> Self::Output {
//         self.to_u16().wrapping_rem(rhs.to_u16())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u16) -> Self::Output {
//         self.to_u16().max(rhs.to_u16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u16) -> Self::Output {
//         self.to_u16().min(rhs.to_u16())
//     }
// }
// impl NormalOut<u32> for u8 {
//     type Output = u32;
//     #[inline(always)]
//     fn _pow(self, rhs: u32) -> Self::Output {
//         self.to_u32().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u32, max: u32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ u32>]();
//             let min = min.[<to_ u32>]();
//             let max = max.[<to_ u32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u32, b: u32) -> Self::Output {
//         self.to_u32().wrapping_mul(a.to_u32()) + b.to_u32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u32) -> Self::Output {
//         self.to_u32().wrapping_add(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u32) -> Self::Output {
//         self.to_u32().wrapping_sub(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u32) -> Self::Output {
//         self.to_u32().wrapping_mul(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u32) -> Self::Output {
//         self.to_u32().wrapping_rem(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u32) -> Self::Output {
//         self.to_u32().max(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u32) -> Self::Output {
//         self.to_u32().min(rhs.to_u32())
//     }
// }
// impl NormalOut<u64> for u8 {
//     type Output = u64;
//     #[inline(always)]
//     fn _pow(self, rhs: u64) -> Self::Output {
//         self.to_u64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u64, max: u64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ u64>]();
//             let min = min.[<to_ u64>]();
//             let max = max.[<to_ u64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u64, b: u64) -> Self::Output {
//         self.to_u64().wrapping_mul(a.to_u64()) + b.to_u64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u64) -> Self::Output {
//         self.to_u64().wrapping_add(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u64) -> Self::Output {
//         self.to_u64().wrapping_sub(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u64) -> Self::Output {
//         self.to_u64().wrapping_mul(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u64) -> Self::Output {
//         self.to_u64().wrapping_rem(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u64) -> Self::Output {
//         self.to_u64().max(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u64) -> Self::Output {
//         self.to_u64().min(rhs.to_u64())
//     }
// }
// impl NormalOut<bf16> for u8 {
//     type Output = bf16;
//     #[inline(always)]
//     fn _pow(self, rhs: bf16) -> Self::Output {
//         self.to_bf16().powf(rhs.to_bf16())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bf16, max: bf16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ bf16>]();
//             let min = min.[<to_ bf16>]();
//             let max = max.[<to_ bf16>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bf16, b: bf16) -> Self::Output {
//         self.to_bf16() * a.to_bf16() + b.to_bf16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bf16) -> Self::Output {
//         self.to_bf16() + rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bf16) -> Self::Output {
//         self.to_bf16() - rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bf16) -> Self::Output {
//         self.to_bf16() * rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bf16) -> Self::Output {
//         self.to_bf16() % rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bf16) -> Self::Output {
//         self.to_bf16().max(rhs.to_bf16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bf16) -> Self::Output {
//         self.to_bf16().min(rhs.to_bf16())
//     }
// }
// impl NormalOut<isize> for u8 {
//     type Output = isize;
//     #[inline(always)]
//     fn _pow(self, rhs: isize) -> Self::Output {
//         self.to_isize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: isize, max: isize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ isize>]();
//             let min = min.[<to_ isize>]();
//             let max = max.[<to_ isize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: isize, b: isize) -> Self::Output {
//         self.to_isize().wrapping_mul(a.to_isize()) + b.to_isize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_add(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_sub(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_mul(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_rem(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: isize) -> Self::Output {
//         self.to_isize().max(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: isize) -> Self::Output {
//         self.to_isize().min(rhs.to_isize())
//     }
// }
// impl NormalOut<usize> for u8 {
//     type Output = usize;
//     #[inline(always)]
//     fn _pow(self, rhs: usize) -> Self::Output {
//         self.to_usize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: usize, max: usize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ usize>]();
//             let min = min.[<to_ usize>]();
//             let max = max.[<to_ usize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: usize, b: usize) -> Self::Output {
//         self.to_usize().wrapping_mul(a.to_usize()) + b.to_usize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: usize) -> Self::Output {
//         self.to_usize().wrapping_add(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: usize) -> Self::Output {
//         self.to_usize().wrapping_sub(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: usize) -> Self::Output {
//         self.to_usize().wrapping_mul(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: usize) -> Self::Output {
//         self.to_usize().wrapping_rem(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: usize) -> Self::Output {
//         self.to_usize().max(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: usize) -> Self::Output {
//         self.to_usize().min(rhs.to_usize())
//     }
// }
// impl NormalOut<Complex32> for u8 {
//     type Output = Complex32;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32().powc(rhs.to_complex32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex32, max: Complex32) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex32>]();
//             let min = min.[<to_ complex32>]();
//             let max = max.[<to_ complex32>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex32::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex32, b: Complex32) -> Self::Output {
//         self.to_complex32() * a.to_complex32() + b.to_complex32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() + rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() - rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() * rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() % rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex32) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex32) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<Complex64> for u8 {
//     type Output = Complex64;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64().powc(rhs.to_complex64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex64, max: Complex64) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex64>]();
//             let min = min.[<to_ complex64>]();
//             let max = max.[<to_ complex64>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex64::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex64, b: Complex64) -> Self::Output {
//         self.to_complex64() * a.to_complex64() + b.to_complex64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() + rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() - rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() * rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() % rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex64) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex64) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<bool> for u16 {
//     type Output = u16;
//     #[inline(always)]
//     fn _pow(self, rhs: bool) -> Self::Output {
//         self.to_u16().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bool, max: bool) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ u16>]();
//             let min = min.[<to_ u16>]();
//             let max = max.[<to_ u16>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bool, b: bool) -> Self::Output {
//         self.to_u16().wrapping_mul(a.to_u16()) + b.to_u16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bool) -> Self::Output {
//         self.to_u16().wrapping_add(rhs.to_u16())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bool) -> Self::Output {
//         self.to_u16().wrapping_sub(rhs.to_u16())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bool) -> Self::Output {
//         self.to_u16().wrapping_mul(rhs.to_u16())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bool) -> Self::Output {
//         self.to_u16().wrapping_rem(rhs.to_u16())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bool) -> Self::Output {
//         self.to_u16().max(rhs.to_u16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bool) -> Self::Output {
//         self.to_u16().min(rhs.to_u16())
//     }
// }
// impl NormalOut<half::f16> for u16 {
//     type Output = f32;
//     #[inline(always)]
//     fn _pow(self, rhs: half::f16) -> Self::Output {
//         self.to_f32().powf(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: half::f16, max: half::f16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f32>]();
//             let min = min.[<to_ f32>]();
//             let max = max.[<to_ f32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: half::f16, b: half::f16) -> Self::Output {
//         self.to_f32() * a.to_f32() + b.to_f32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: half::f16) -> Self::Output {
//         self.to_f32() + rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: half::f16) -> Self::Output {
//         self.to_f32() - rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: half::f16) -> Self::Output {
//         self.to_f32() * rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: half::f16) -> Self::Output {
//         self.to_f32() % rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: half::f16) -> Self::Output {
//         self.to_f32().max(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: half::f16) -> Self::Output {
//         self.to_f32().min(rhs.to_f32())
//     }
// }
// impl NormalOut<f32> for u16 {
//     type Output = f32;
//     #[inline(always)]
//     fn _pow(self, rhs: f32) -> Self::Output {
//         self.to_f32().powf(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f32, max: f32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f32>]();
//             let min = min.[<to_ f32>]();
//             let max = max.[<to_ f32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f32, b: f32) -> Self::Output {
//         self.to_f32() * a.to_f32() + b.to_f32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f32) -> Self::Output {
//         self.to_f32() + rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f32) -> Self::Output {
//         self.to_f32() - rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f32) -> Self::Output {
//         self.to_f32() * rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f32) -> Self::Output {
//         self.to_f32() % rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f32) -> Self::Output {
//         self.to_f32().max(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f32) -> Self::Output {
//         self.to_f32().min(rhs.to_f32())
//     }
// }
// impl NormalOut<f64> for u16 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: f64) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f64, max: f64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f64, b: f64) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f64) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f64) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f64) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f64) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f64) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f64) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<i8> for u16 {
//     type Output = u16;
//     #[inline(always)]
//     fn _pow(self, rhs: i8) -> Self::Output {
//         self.to_u16().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i8, max: i8) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ u16>]();
//             let min = min.[<to_ u16>]();
//             let max = max.[<to_ u16>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i8, b: i8) -> Self::Output {
//         self.to_u16().wrapping_mul(a.to_u16()) + b.to_u16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i8) -> Self::Output {
//         self.to_u16().wrapping_add(rhs.to_u16())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i8) -> Self::Output {
//         self.to_u16().wrapping_sub(rhs.to_u16())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i8) -> Self::Output {
//         self.to_u16().wrapping_mul(rhs.to_u16())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i8) -> Self::Output {
//         self.to_u16().wrapping_rem(rhs.to_u16())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i8) -> Self::Output {
//         self.to_u16().max(rhs.to_u16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i8) -> Self::Output {
//         self.to_u16().min(rhs.to_u16())
//     }
// }
// impl NormalOut<i16> for u16 {
//     type Output = u16;
//     #[inline(always)]
//     fn _pow(self, rhs: i16) -> Self::Output {
//         self.to_u16().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i16, max: i16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ u16>]();
//             let min = min.[<to_ u16>]();
//             let max = max.[<to_ u16>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i16, b: i16) -> Self::Output {
//         self.to_u16().wrapping_mul(a.to_u16()) + b.to_u16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i16) -> Self::Output {
//         self.to_u16().wrapping_add(rhs.to_u16())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i16) -> Self::Output {
//         self.to_u16().wrapping_sub(rhs.to_u16())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i16) -> Self::Output {
//         self.to_u16().wrapping_mul(rhs.to_u16())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i16) -> Self::Output {
//         self.to_u16().wrapping_rem(rhs.to_u16())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i16) -> Self::Output {
//         self.to_u16().max(rhs.to_u16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i16) -> Self::Output {
//         self.to_u16().min(rhs.to_u16())
//     }
// }
// impl NormalOut<i32> for u16 {
//     type Output = i32;
//     #[inline(always)]
//     fn _pow(self, rhs: i32) -> Self::Output {
//         self.to_i32().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i32, max: i32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i32>]();
//             let min = min.[<to_ i32>]();
//             let max = max.[<to_ i32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i32, b: i32) -> Self::Output {
//         self.to_i32().wrapping_mul(a.to_i32()) + b.to_i32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i32) -> Self::Output {
//         self.to_i32().wrapping_add(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i32) -> Self::Output {
//         self.to_i32().wrapping_sub(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i32) -> Self::Output {
//         self.to_i32().wrapping_mul(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i32) -> Self::Output {
//         self.to_i32().wrapping_rem(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i32) -> Self::Output {
//         self.to_i32().max(rhs.to_i32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i32) -> Self::Output {
//         self.to_i32().min(rhs.to_i32())
//     }
// }
// impl NormalOut<i64> for u16 {
//     type Output = i64;
//     #[inline(always)]
//     fn _pow(self, rhs: i64) -> Self::Output {
//         self.to_i64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i64, max: i64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i64>]();
//             let min = min.[<to_ i64>]();
//             let max = max.[<to_ i64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i64, b: i64) -> Self::Output {
//         self.to_i64().wrapping_mul(a.to_i64()) + b.to_i64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_add(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_sub(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_mul(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_rem(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i64) -> Self::Output {
//         self.to_i64().max(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i64) -> Self::Output {
//         self.to_i64().min(rhs.to_i64())
//     }
// }
// impl NormalOut<u8> for u16 {
//     type Output = u16;
//     #[inline(always)]
//     fn _pow(self, rhs: u8) -> Self::Output {
//         self.to_u16().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u8, max: u8) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ u16>]();
//             let min = min.[<to_ u16>]();
//             let max = max.[<to_ u16>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u8, b: u8) -> Self::Output {
//         self.to_u16().wrapping_mul(a.to_u16()) + b.to_u16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u8) -> Self::Output {
//         self.to_u16().wrapping_add(rhs.to_u16())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u8) -> Self::Output {
//         self.to_u16().wrapping_sub(rhs.to_u16())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u8) -> Self::Output {
//         self.to_u16().wrapping_mul(rhs.to_u16())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u8) -> Self::Output {
//         self.to_u16().wrapping_rem(rhs.to_u16())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u8) -> Self::Output {
//         self.to_u16().max(rhs.to_u16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u8) -> Self::Output {
//         self.to_u16().min(rhs.to_u16())
//     }
// }
// impl NormalOut<u16> for u16 {
//     type Output = u16;
//     #[inline(always)]
//     fn _pow(self, rhs: u16) -> Self::Output {
//         self.to_u16().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u16, max: u16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ u16>]();
//             let min = min.[<to_ u16>]();
//             let max = max.[<to_ u16>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u16, b: u16) -> Self::Output {
//         self.to_u16().wrapping_mul(a.to_u16()) + b.to_u16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u16) -> Self::Output {
//         self.to_u16().wrapping_add(rhs.to_u16())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u16) -> Self::Output {
//         self.to_u16().wrapping_sub(rhs.to_u16())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u16) -> Self::Output {
//         self.to_u16().wrapping_mul(rhs.to_u16())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u16) -> Self::Output {
//         self.to_u16().wrapping_rem(rhs.to_u16())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u16) -> Self::Output {
//         self.to_u16().max(rhs.to_u16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u16) -> Self::Output {
//         self.to_u16().min(rhs.to_u16())
//     }
// }
// impl NormalOut<u32> for u16 {
//     type Output = u32;
//     #[inline(always)]
//     fn _pow(self, rhs: u32) -> Self::Output {
//         self.to_u32().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u32, max: u32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ u32>]();
//             let min = min.[<to_ u32>]();
//             let max = max.[<to_ u32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u32, b: u32) -> Self::Output {
//         self.to_u32().wrapping_mul(a.to_u32()) + b.to_u32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u32) -> Self::Output {
//         self.to_u32().wrapping_add(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u32) -> Self::Output {
//         self.to_u32().wrapping_sub(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u32) -> Self::Output {
//         self.to_u32().wrapping_mul(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u32) -> Self::Output {
//         self.to_u32().wrapping_rem(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u32) -> Self::Output {
//         self.to_u32().max(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u32) -> Self::Output {
//         self.to_u32().min(rhs.to_u32())
//     }
// }
// impl NormalOut<u64> for u16 {
//     type Output = u64;
//     #[inline(always)]
//     fn _pow(self, rhs: u64) -> Self::Output {
//         self.to_u64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u64, max: u64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ u64>]();
//             let min = min.[<to_ u64>]();
//             let max = max.[<to_ u64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u64, b: u64) -> Self::Output {
//         self.to_u64().wrapping_mul(a.to_u64()) + b.to_u64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u64) -> Self::Output {
//         self.to_u64().wrapping_add(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u64) -> Self::Output {
//         self.to_u64().wrapping_sub(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u64) -> Self::Output {
//         self.to_u64().wrapping_mul(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u64) -> Self::Output {
//         self.to_u64().wrapping_rem(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u64) -> Self::Output {
//         self.to_u64().max(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u64) -> Self::Output {
//         self.to_u64().min(rhs.to_u64())
//     }
// }
// impl NormalOut<bf16> for u16 {
//     type Output = f32;
//     #[inline(always)]
//     fn _pow(self, rhs: bf16) -> Self::Output {
//         self.to_f32().powf(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bf16, max: bf16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f32>]();
//             let min = min.[<to_ f32>]();
//             let max = max.[<to_ f32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bf16, b: bf16) -> Self::Output {
//         self.to_f32() * a.to_f32() + b.to_f32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bf16) -> Self::Output {
//         self.to_f32() + rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bf16) -> Self::Output {
//         self.to_f32() - rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bf16) -> Self::Output {
//         self.to_f32() * rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bf16) -> Self::Output {
//         self.to_f32() % rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bf16) -> Self::Output {
//         self.to_f32().max(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bf16) -> Self::Output {
//         self.to_f32().min(rhs.to_f32())
//     }
// }
// impl NormalOut<isize> for u16 {
//     type Output = isize;
//     #[inline(always)]
//     fn _pow(self, rhs: isize) -> Self::Output {
//         self.to_isize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: isize, max: isize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ isize>]();
//             let min = min.[<to_ isize>]();
//             let max = max.[<to_ isize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: isize, b: isize) -> Self::Output {
//         self.to_isize().wrapping_mul(a.to_isize()) + b.to_isize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_add(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_sub(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_mul(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_rem(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: isize) -> Self::Output {
//         self.to_isize().max(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: isize) -> Self::Output {
//         self.to_isize().min(rhs.to_isize())
//     }
// }
// impl NormalOut<usize> for u16 {
//     type Output = usize;
//     #[inline(always)]
//     fn _pow(self, rhs: usize) -> Self::Output {
//         self.to_usize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: usize, max: usize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ usize>]();
//             let min = min.[<to_ usize>]();
//             let max = max.[<to_ usize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: usize, b: usize) -> Self::Output {
//         self.to_usize().wrapping_mul(a.to_usize()) + b.to_usize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: usize) -> Self::Output {
//         self.to_usize().wrapping_add(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: usize) -> Self::Output {
//         self.to_usize().wrapping_sub(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: usize) -> Self::Output {
//         self.to_usize().wrapping_mul(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: usize) -> Self::Output {
//         self.to_usize().wrapping_rem(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: usize) -> Self::Output {
//         self.to_usize().max(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: usize) -> Self::Output {
//         self.to_usize().min(rhs.to_usize())
//     }
// }
// impl NormalOut<Complex32> for u16 {
//     type Output = Complex32;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32().powc(rhs.to_complex32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex32, max: Complex32) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex32>]();
//             let min = min.[<to_ complex32>]();
//             let max = max.[<to_ complex32>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex32::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex32, b: Complex32) -> Self::Output {
//         self.to_complex32() * a.to_complex32() + b.to_complex32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() + rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() - rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() * rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() % rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex32) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex32) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<Complex64> for u16 {
//     type Output = Complex64;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64().powc(rhs.to_complex64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex64, max: Complex64) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex64>]();
//             let min = min.[<to_ complex64>]();
//             let max = max.[<to_ complex64>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex64::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex64, b: Complex64) -> Self::Output {
//         self.to_complex64() * a.to_complex64() + b.to_complex64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() + rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() - rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() * rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() % rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex64) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex64) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<bool> for u32 {
//     type Output = u32;
//     #[inline(always)]
//     fn _pow(self, rhs: bool) -> Self::Output {
//         self.to_u32().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bool, max: bool) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ u32>]();
//             let min = min.[<to_ u32>]();
//             let max = max.[<to_ u32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bool, b: bool) -> Self::Output {
//         self.to_u32().wrapping_mul(a.to_u32()) + b.to_u32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bool) -> Self::Output {
//         self.to_u32().wrapping_add(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bool) -> Self::Output {
//         self.to_u32().wrapping_sub(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bool) -> Self::Output {
//         self.to_u32().wrapping_mul(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bool) -> Self::Output {
//         self.to_u32().wrapping_rem(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bool) -> Self::Output {
//         self.to_u32().max(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bool) -> Self::Output {
//         self.to_u32().min(rhs.to_u32())
//     }
// }
// impl NormalOut<half::f16> for u32 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: half::f16) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: half::f16, max: half::f16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: half::f16, b: half::f16) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: half::f16) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: half::f16) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: half::f16) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: half::f16) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: half::f16) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: half::f16) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<f32> for u32 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: f32) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f32, max: f32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f32, b: f32) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f32) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f32) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f32) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f32) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f32) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f32) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<f64> for u32 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: f64) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f64, max: f64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f64, b: f64) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f64) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f64) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f64) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f64) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f64) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f64) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<i8> for u32 {
//     type Output = u32;
//     #[inline(always)]
//     fn _pow(self, rhs: i8) -> Self::Output {
//         self.to_u32().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i8, max: i8) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ u32>]();
//             let min = min.[<to_ u32>]();
//             let max = max.[<to_ u32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i8, b: i8) -> Self::Output {
//         self.to_u32().wrapping_mul(a.to_u32()) + b.to_u32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i8) -> Self::Output {
//         self.to_u32().wrapping_add(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i8) -> Self::Output {
//         self.to_u32().wrapping_sub(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i8) -> Self::Output {
//         self.to_u32().wrapping_mul(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i8) -> Self::Output {
//         self.to_u32().wrapping_rem(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i8) -> Self::Output {
//         self.to_u32().max(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i8) -> Self::Output {
//         self.to_u32().min(rhs.to_u32())
//     }
// }
// impl NormalOut<i16> for u32 {
//     type Output = u32;
//     #[inline(always)]
//     fn _pow(self, rhs: i16) -> Self::Output {
//         self.to_u32().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i16, max: i16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ u32>]();
//             let min = min.[<to_ u32>]();
//             let max = max.[<to_ u32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i16, b: i16) -> Self::Output {
//         self.to_u32().wrapping_mul(a.to_u32()) + b.to_u32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i16) -> Self::Output {
//         self.to_u32().wrapping_add(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i16) -> Self::Output {
//         self.to_u32().wrapping_sub(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i16) -> Self::Output {
//         self.to_u32().wrapping_mul(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i16) -> Self::Output {
//         self.to_u32().wrapping_rem(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i16) -> Self::Output {
//         self.to_u32().max(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i16) -> Self::Output {
//         self.to_u32().min(rhs.to_u32())
//     }
// }
// impl NormalOut<i32> for u32 {
//     type Output = u32;
//     #[inline(always)]
//     fn _pow(self, rhs: i32) -> Self::Output {
//         self.to_u32().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i32, max: i32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ u32>]();
//             let min = min.[<to_ u32>]();
//             let max = max.[<to_ u32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i32, b: i32) -> Self::Output {
//         self.to_u32().wrapping_mul(a.to_u32()) + b.to_u32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i32) -> Self::Output {
//         self.to_u32().wrapping_add(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i32) -> Self::Output {
//         self.to_u32().wrapping_sub(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i32) -> Self::Output {
//         self.to_u32().wrapping_mul(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i32) -> Self::Output {
//         self.to_u32().wrapping_rem(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i32) -> Self::Output {
//         self.to_u32().max(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i32) -> Self::Output {
//         self.to_u32().min(rhs.to_u32())
//     }
// }
// impl NormalOut<i64> for u32 {
//     type Output = i64;
//     #[inline(always)]
//     fn _pow(self, rhs: i64) -> Self::Output {
//         self.to_i64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i64, max: i64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i64>]();
//             let min = min.[<to_ i64>]();
//             let max = max.[<to_ i64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i64, b: i64) -> Self::Output {
//         self.to_i64().wrapping_mul(a.to_i64()) + b.to_i64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_add(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_sub(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_mul(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_rem(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i64) -> Self::Output {
//         self.to_i64().max(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i64) -> Self::Output {
//         self.to_i64().min(rhs.to_i64())
//     }
// }
// impl NormalOut<u8> for u32 {
//     type Output = u32;
//     #[inline(always)]
//     fn _pow(self, rhs: u8) -> Self::Output {
//         self.to_u32().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u8, max: u8) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ u32>]();
//             let min = min.[<to_ u32>]();
//             let max = max.[<to_ u32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u8, b: u8) -> Self::Output {
//         self.to_u32().wrapping_mul(a.to_u32()) + b.to_u32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u8) -> Self::Output {
//         self.to_u32().wrapping_add(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u8) -> Self::Output {
//         self.to_u32().wrapping_sub(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u8) -> Self::Output {
//         self.to_u32().wrapping_mul(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u8) -> Self::Output {
//         self.to_u32().wrapping_rem(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u8) -> Self::Output {
//         self.to_u32().max(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u8) -> Self::Output {
//         self.to_u32().min(rhs.to_u32())
//     }
// }
// impl NormalOut<u16> for u32 {
//     type Output = u32;
//     #[inline(always)]
//     fn _pow(self, rhs: u16) -> Self::Output {
//         self.to_u32().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u16, max: u16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ u32>]();
//             let min = min.[<to_ u32>]();
//             let max = max.[<to_ u32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u16, b: u16) -> Self::Output {
//         self.to_u32().wrapping_mul(a.to_u32()) + b.to_u32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u16) -> Self::Output {
//         self.to_u32().wrapping_add(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u16) -> Self::Output {
//         self.to_u32().wrapping_sub(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u16) -> Self::Output {
//         self.to_u32().wrapping_mul(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u16) -> Self::Output {
//         self.to_u32().wrapping_rem(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u16) -> Self::Output {
//         self.to_u32().max(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u16) -> Self::Output {
//         self.to_u32().min(rhs.to_u32())
//     }
// }
// impl NormalOut<u32> for u32 {
//     type Output = u32;
//     #[inline(always)]
//     fn _pow(self, rhs: u32) -> Self::Output {
//         self.to_u32().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u32, max: u32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ u32>]();
//             let min = min.[<to_ u32>]();
//             let max = max.[<to_ u32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u32, b: u32) -> Self::Output {
//         self.to_u32().wrapping_mul(a.to_u32()) + b.to_u32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u32) -> Self::Output {
//         self.to_u32().wrapping_add(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u32) -> Self::Output {
//         self.to_u32().wrapping_sub(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u32) -> Self::Output {
//         self.to_u32().wrapping_mul(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u32) -> Self::Output {
//         self.to_u32().wrapping_rem(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u32) -> Self::Output {
//         self.to_u32().max(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u32) -> Self::Output {
//         self.to_u32().min(rhs.to_u32())
//     }
// }
// impl NormalOut<u64> for u32 {
//     type Output = u64;
//     #[inline(always)]
//     fn _pow(self, rhs: u64) -> Self::Output {
//         self.to_u64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u64, max: u64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ u64>]();
//             let min = min.[<to_ u64>]();
//             let max = max.[<to_ u64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u64, b: u64) -> Self::Output {
//         self.to_u64().wrapping_mul(a.to_u64()) + b.to_u64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u64) -> Self::Output {
//         self.to_u64().wrapping_add(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u64) -> Self::Output {
//         self.to_u64().wrapping_sub(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u64) -> Self::Output {
//         self.to_u64().wrapping_mul(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u64) -> Self::Output {
//         self.to_u64().wrapping_rem(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u64) -> Self::Output {
//         self.to_u64().max(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u64) -> Self::Output {
//         self.to_u64().min(rhs.to_u64())
//     }
// }
// impl NormalOut<bf16> for u32 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: bf16) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bf16, max: bf16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bf16, b: bf16) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bf16) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bf16) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bf16) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bf16) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bf16) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bf16) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<isize> for u32 {
//     type Output = isize;
//     #[inline(always)]
//     fn _pow(self, rhs: isize) -> Self::Output {
//         self.to_isize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: isize, max: isize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ isize>]();
//             let min = min.[<to_ isize>]();
//             let max = max.[<to_ isize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: isize, b: isize) -> Self::Output {
//         self.to_isize().wrapping_mul(a.to_isize()) + b.to_isize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_add(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_sub(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_mul(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_rem(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: isize) -> Self::Output {
//         self.to_isize().max(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: isize) -> Self::Output {
//         self.to_isize().min(rhs.to_isize())
//     }
// }
// impl NormalOut<usize> for u32 {
//     type Output = usize;
//     #[inline(always)]
//     fn _pow(self, rhs: usize) -> Self::Output {
//         self.to_usize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: usize, max: usize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ usize>]();
//             let min = min.[<to_ usize>]();
//             let max = max.[<to_ usize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: usize, b: usize) -> Self::Output {
//         self.to_usize().wrapping_mul(a.to_usize()) + b.to_usize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: usize) -> Self::Output {
//         self.to_usize().wrapping_add(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: usize) -> Self::Output {
//         self.to_usize().wrapping_sub(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: usize) -> Self::Output {
//         self.to_usize().wrapping_mul(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: usize) -> Self::Output {
//         self.to_usize().wrapping_rem(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: usize) -> Self::Output {
//         self.to_usize().max(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: usize) -> Self::Output {
//         self.to_usize().min(rhs.to_usize())
//     }
// }
// impl NormalOut<Complex32> for u32 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex32) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex32, max: Complex32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex32, b: Complex32) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex32) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex32) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex32) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex32) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex32) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex32) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<Complex64> for u32 {
//     type Output = Complex64;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64().powc(rhs.to_complex64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex64, max: Complex64) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex64>]();
//             let min = min.[<to_ complex64>]();
//             let max = max.[<to_ complex64>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex64::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex64, b: Complex64) -> Self::Output {
//         self.to_complex64() * a.to_complex64() + b.to_complex64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() + rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() - rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() * rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() % rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex64) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex64) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<bool> for u64 {
//     type Output = u64;
//     #[inline(always)]
//     fn _pow(self, rhs: bool) -> Self::Output {
//         self.to_u64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bool, max: bool) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ u64>]();
//             let min = min.[<to_ u64>]();
//             let max = max.[<to_ u64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bool, b: bool) -> Self::Output {
//         self.to_u64().wrapping_mul(a.to_u64()) + b.to_u64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bool) -> Self::Output {
//         self.to_u64().wrapping_add(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bool) -> Self::Output {
//         self.to_u64().wrapping_sub(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bool) -> Self::Output {
//         self.to_u64().wrapping_mul(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bool) -> Self::Output {
//         self.to_u64().wrapping_rem(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bool) -> Self::Output {
//         self.to_u64().max(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bool) -> Self::Output {
//         self.to_u64().min(rhs.to_u64())
//     }
// }
// impl NormalOut<half::f16> for u64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: half::f16) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: half::f16, max: half::f16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: half::f16, b: half::f16) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: half::f16) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: half::f16) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: half::f16) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: half::f16) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: half::f16) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: half::f16) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<f32> for u64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: f32) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f32, max: f32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f32, b: f32) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f32) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f32) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f32) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f32) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f32) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f32) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<f64> for u64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: f64) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f64, max: f64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f64, b: f64) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f64) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f64) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f64) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f64) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f64) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f64) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<i8> for u64 {
//     type Output = u64;
//     #[inline(always)]
//     fn _pow(self, rhs: i8) -> Self::Output {
//         self.to_u64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i8, max: i8) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ u64>]();
//             let min = min.[<to_ u64>]();
//             let max = max.[<to_ u64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i8, b: i8) -> Self::Output {
//         self.to_u64().wrapping_mul(a.to_u64()) + b.to_u64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i8) -> Self::Output {
//         self.to_u64().wrapping_add(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i8) -> Self::Output {
//         self.to_u64().wrapping_sub(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i8) -> Self::Output {
//         self.to_u64().wrapping_mul(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i8) -> Self::Output {
//         self.to_u64().wrapping_rem(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i8) -> Self::Output {
//         self.to_u64().max(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i8) -> Self::Output {
//         self.to_u64().min(rhs.to_u64())
//     }
// }
// impl NormalOut<i16> for u64 {
//     type Output = u64;
//     #[inline(always)]
//     fn _pow(self, rhs: i16) -> Self::Output {
//         self.to_u64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i16, max: i16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ u64>]();
//             let min = min.[<to_ u64>]();
//             let max = max.[<to_ u64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i16, b: i16) -> Self::Output {
//         self.to_u64().wrapping_mul(a.to_u64()) + b.to_u64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i16) -> Self::Output {
//         self.to_u64().wrapping_add(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i16) -> Self::Output {
//         self.to_u64().wrapping_sub(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i16) -> Self::Output {
//         self.to_u64().wrapping_mul(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i16) -> Self::Output {
//         self.to_u64().wrapping_rem(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i16) -> Self::Output {
//         self.to_u64().max(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i16) -> Self::Output {
//         self.to_u64().min(rhs.to_u64())
//     }
// }
// impl NormalOut<i32> for u64 {
//     type Output = u64;
//     #[inline(always)]
//     fn _pow(self, rhs: i32) -> Self::Output {
//         self.to_u64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i32, max: i32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ u64>]();
//             let min = min.[<to_ u64>]();
//             let max = max.[<to_ u64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i32, b: i32) -> Self::Output {
//         self.to_u64().wrapping_mul(a.to_u64()) + b.to_u64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i32) -> Self::Output {
//         self.to_u64().wrapping_add(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i32) -> Self::Output {
//         self.to_u64().wrapping_sub(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i32) -> Self::Output {
//         self.to_u64().wrapping_mul(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i32) -> Self::Output {
//         self.to_u64().wrapping_rem(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i32) -> Self::Output {
//         self.to_u64().max(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i32) -> Self::Output {
//         self.to_u64().min(rhs.to_u64())
//     }
// }
// impl NormalOut<i64> for u64 {
//     type Output = i64;
//     #[inline(always)]
//     fn _pow(self, rhs: i64) -> Self::Output {
//         self.to_i64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i64, max: i64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ i64>]();
//             let min = min.[<to_ i64>]();
//             let max = max.[<to_ i64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i64, b: i64) -> Self::Output {
//         self.to_i64().wrapping_mul(a.to_i64()) + b.to_i64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_add(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_sub(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_mul(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i64) -> Self::Output {
//         self.to_i64().wrapping_rem(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i64) -> Self::Output {
//         self.to_i64().max(rhs.to_i64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i64) -> Self::Output {
//         self.to_i64().min(rhs.to_i64())
//     }
// }
// impl NormalOut<u8> for u64 {
//     type Output = u64;
//     #[inline(always)]
//     fn _pow(self, rhs: u8) -> Self::Output {
//         self.to_u64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u8, max: u8) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ u64>]();
//             let min = min.[<to_ u64>]();
//             let max = max.[<to_ u64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u8, b: u8) -> Self::Output {
//         self.to_u64().wrapping_mul(a.to_u64()) + b.to_u64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u8) -> Self::Output {
//         self.to_u64().wrapping_add(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u8) -> Self::Output {
//         self.to_u64().wrapping_sub(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u8) -> Self::Output {
//         self.to_u64().wrapping_mul(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u8) -> Self::Output {
//         self.to_u64().wrapping_rem(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u8) -> Self::Output {
//         self.to_u64().max(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u8) -> Self::Output {
//         self.to_u64().min(rhs.to_u64())
//     }
// }
// impl NormalOut<u16> for u64 {
//     type Output = u64;
//     #[inline(always)]
//     fn _pow(self, rhs: u16) -> Self::Output {
//         self.to_u64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u16, max: u16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ u64>]();
//             let min = min.[<to_ u64>]();
//             let max = max.[<to_ u64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u16, b: u16) -> Self::Output {
//         self.to_u64().wrapping_mul(a.to_u64()) + b.to_u64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u16) -> Self::Output {
//         self.to_u64().wrapping_add(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u16) -> Self::Output {
//         self.to_u64().wrapping_sub(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u16) -> Self::Output {
//         self.to_u64().wrapping_mul(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u16) -> Self::Output {
//         self.to_u64().wrapping_rem(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u16) -> Self::Output {
//         self.to_u64().max(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u16) -> Self::Output {
//         self.to_u64().min(rhs.to_u64())
//     }
// }
// impl NormalOut<u32> for u64 {
//     type Output = u64;
//     #[inline(always)]
//     fn _pow(self, rhs: u32) -> Self::Output {
//         self.to_u64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u32, max: u32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ u64>]();
//             let min = min.[<to_ u64>]();
//             let max = max.[<to_ u64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u32, b: u32) -> Self::Output {
//         self.to_u64().wrapping_mul(a.to_u64()) + b.to_u64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u32) -> Self::Output {
//         self.to_u64().wrapping_add(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u32) -> Self::Output {
//         self.to_u64().wrapping_sub(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u32) -> Self::Output {
//         self.to_u64().wrapping_mul(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u32) -> Self::Output {
//         self.to_u64().wrapping_rem(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u32) -> Self::Output {
//         self.to_u64().max(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u32) -> Self::Output {
//         self.to_u64().min(rhs.to_u64())
//     }
// }
// impl NormalOut<u64> for u64 {
//     type Output = u64;
//     #[inline(always)]
//     fn _pow(self, rhs: u64) -> Self::Output {
//         self.to_u64().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u64, max: u64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ u64>]();
//             let min = min.[<to_ u64>]();
//             let max = max.[<to_ u64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u64, b: u64) -> Self::Output {
//         self.to_u64().wrapping_mul(a.to_u64()) + b.to_u64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u64) -> Self::Output {
//         self.to_u64().wrapping_add(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u64) -> Self::Output {
//         self.to_u64().wrapping_sub(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u64) -> Self::Output {
//         self.to_u64().wrapping_mul(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u64) -> Self::Output {
//         self.to_u64().wrapping_rem(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u64) -> Self::Output {
//         self.to_u64().max(rhs.to_u64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u64) -> Self::Output {
//         self.to_u64().min(rhs.to_u64())
//     }
// }
// impl NormalOut<bf16> for u64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: bf16) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bf16, max: bf16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bf16, b: bf16) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bf16) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bf16) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bf16) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bf16) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bf16) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bf16) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<isize> for u64 {
//     type Output = isize;
//     #[inline(always)]
//     fn _pow(self, rhs: isize) -> Self::Output {
//         self.to_isize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: isize, max: isize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ isize>]();
//             let min = min.[<to_ isize>]();
//             let max = max.[<to_ isize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: isize, b: isize) -> Self::Output {
//         self.to_isize().wrapping_mul(a.to_isize()) + b.to_isize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_add(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_sub(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_mul(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_rem(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: isize) -> Self::Output {
//         self.to_isize().max(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: isize) -> Self::Output {
//         self.to_isize().min(rhs.to_isize())
//     }
// }
// impl NormalOut<usize> for u64 {
//     type Output = usize;
//     #[inline(always)]
//     fn _pow(self, rhs: usize) -> Self::Output {
//         self.to_usize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: usize, max: usize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ usize>]();
//             let min = min.[<to_ usize>]();
//             let max = max.[<to_ usize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: usize, b: usize) -> Self::Output {
//         self.to_usize().wrapping_mul(a.to_usize()) + b.to_usize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: usize) -> Self::Output {
//         self.to_usize().wrapping_add(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: usize) -> Self::Output {
//         self.to_usize().wrapping_sub(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: usize) -> Self::Output {
//         self.to_usize().wrapping_mul(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: usize) -> Self::Output {
//         self.to_usize().wrapping_rem(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: usize) -> Self::Output {
//         self.to_usize().max(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: usize) -> Self::Output {
//         self.to_usize().min(rhs.to_usize())
//     }
// }
// impl NormalOut<Complex32> for u64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex32) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex32, max: Complex32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex32, b: Complex32) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex32) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex32) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex32) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex32) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex32) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex32) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<Complex64> for u64 {
//     type Output = Complex64;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64().powc(rhs.to_complex64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex64, max: Complex64) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex64>]();
//             let min = min.[<to_ complex64>]();
//             let max = max.[<to_ complex64>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex64::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex64, b: Complex64) -> Self::Output {
//         self.to_complex64() * a.to_complex64() + b.to_complex64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() + rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() - rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() * rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() % rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex64) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex64) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<bool> for bf16 {
//     type Output = bf16;
//     #[inline(always)]
//     fn _pow(self, rhs: bool) -> Self::Output {
//         self.to_bf16().powf(rhs.to_bf16())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bool, max: bool) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ bf16>]();
//             let min = min.[<to_ bf16>]();
//             let max = max.[<to_ bf16>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bool, b: bool) -> Self::Output {
//         self.to_bf16() * a.to_bf16() + b.to_bf16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bool) -> Self::Output {
//         self.to_bf16() + rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bool) -> Self::Output {
//         self.to_bf16() - rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bool) -> Self::Output {
//         self.to_bf16() * rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bool) -> Self::Output {
//         self.to_bf16() % rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bool) -> Self::Output {
//         self.to_bf16().max(rhs.to_bf16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bool) -> Self::Output {
//         self.to_bf16().min(rhs.to_bf16())
//     }
// }
// impl NormalOut<half::f16> for bf16 {
//     type Output = half::f16;
//     #[inline(always)]
//     fn _pow(self, rhs: half::f16) -> Self::Output {
//         self.to_f16().powf(rhs.to_f16())
//     }
//     #[inline(always)]
//     fn _clip(self, min: half::f16, max: half::f16) -> Self::Output {
//         paste::paste! {
//             let a = self.to_f16();
//             let min = min.to_f16();
//             let max = max.to_f16();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: half::f16, b: half::f16) -> Self::Output {
//         self.to_f16() * a.to_f16() + b.to_f16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: half::f16) -> Self::Output {
//         self.to_f16() + rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: half::f16) -> Self::Output {
//         self.to_f16() - rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: half::f16) -> Self::Output {
//         self.to_f16() * rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: half::f16) -> Self::Output {
//         self.to_f16() % rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: half::f16) -> Self::Output {
//         self.to_f16().max(rhs.to_f16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: half::f16) -> Self::Output {
//         self.to_f16().min(rhs.to_f16())
//     }
// }
// impl NormalOut<f32> for bf16 {
//     type Output = f32;
//     #[inline(always)]
//     fn _pow(self, rhs: f32) -> Self::Output {
//         self.to_f32().powf(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f32, max: f32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f32>]();
//             let min = min.[<to_ f32>]();
//             let max = max.[<to_ f32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f32, b: f32) -> Self::Output {
//         self.to_f32() * a.to_f32() + b.to_f32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f32) -> Self::Output {
//         self.to_f32() + rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f32) -> Self::Output {
//         self.to_f32() - rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f32) -> Self::Output {
//         self.to_f32() * rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f32) -> Self::Output {
//         self.to_f32() % rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f32) -> Self::Output {
//         self.to_f32().max(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f32) -> Self::Output {
//         self.to_f32().min(rhs.to_f32())
//     }
// }
// impl NormalOut<f64> for bf16 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: f64) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f64, max: f64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f64, b: f64) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f64) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f64) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f64) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f64) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f64) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f64) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<i8> for bf16 {
//     type Output = bf16;
//     #[inline(always)]
//     fn _pow(self, rhs: i8) -> Self::Output {
//         self.to_bf16().powf(rhs.to_bf16())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i8, max: i8) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ bf16>]();
//             let min = min.[<to_ bf16>]();
//             let max = max.[<to_ bf16>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i8, b: i8) -> Self::Output {
//         self.to_bf16() * a.to_bf16() + b.to_bf16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i8) -> Self::Output {
//         self.to_bf16() + rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i8) -> Self::Output {
//         self.to_bf16() - rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i8) -> Self::Output {
//         self.to_bf16() * rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i8) -> Self::Output {
//         self.to_bf16() % rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i8) -> Self::Output {
//         self.to_bf16().max(rhs.to_bf16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i8) -> Self::Output {
//         self.to_bf16().min(rhs.to_bf16())
//     }
// }
// impl NormalOut<i16> for bf16 {
//     type Output = half::f16;
//     #[inline(always)]
//     fn _pow(self, rhs: i16) -> Self::Output {
//         self.to_f16().powf(rhs.to_f16())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i16, max: i16) -> Self::Output {
//         paste::paste! {
//             let a = self.to_f16();
//             let min = min.to_f16();
//             let max = max.to_f16();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i16, b: i16) -> Self::Output {
//         self.to_f16() * a.to_f16() + b.to_f16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i16) -> Self::Output {
//         self.to_f16() + rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i16) -> Self::Output {
//         self.to_f16() - rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i16) -> Self::Output {
//         self.to_f16() * rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i16) -> Self::Output {
//         self.to_f16() % rhs.to_f16()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i16) -> Self::Output {
//         self.to_f16().max(rhs.to_f16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i16) -> Self::Output {
//         self.to_f16().min(rhs.to_f16())
//     }
// }
// impl NormalOut<i32> for bf16 {
//     type Output = f32;
//     #[inline(always)]
//     fn _pow(self, rhs: i32) -> Self::Output {
//         self.to_f32().powf(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i32, max: i32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f32>]();
//             let min = min.[<to_ f32>]();
//             let max = max.[<to_ f32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i32, b: i32) -> Self::Output {
//         self.to_f32() * a.to_f32() + b.to_f32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i32) -> Self::Output {
//         self.to_f32() + rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i32) -> Self::Output {
//         self.to_f32() - rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i32) -> Self::Output {
//         self.to_f32() * rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i32) -> Self::Output {
//         self.to_f32() % rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i32) -> Self::Output {
//         self.to_f32().max(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i32) -> Self::Output {
//         self.to_f32().min(rhs.to_f32())
//     }
// }
// impl NormalOut<i64> for bf16 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: i64) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i64, max: i64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i64, b: i64) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i64) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i64) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i64) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i64) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i64) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i64) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<u8> for bf16 {
//     type Output = bf16;
//     #[inline(always)]
//     fn _pow(self, rhs: u8) -> Self::Output {
//         self.to_bf16().powf(rhs.to_bf16())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u8, max: u8) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ bf16>]();
//             let min = min.[<to_ bf16>]();
//             let max = max.[<to_ bf16>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u8, b: u8) -> Self::Output {
//         self.to_bf16() * a.to_bf16() + b.to_bf16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u8) -> Self::Output {
//         self.to_bf16() + rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u8) -> Self::Output {
//         self.to_bf16() - rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u8) -> Self::Output {
//         self.to_bf16() * rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u8) -> Self::Output {
//         self.to_bf16() % rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u8) -> Self::Output {
//         self.to_bf16().max(rhs.to_bf16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u8) -> Self::Output {
//         self.to_bf16().min(rhs.to_bf16())
//     }
// }
// impl NormalOut<u16> for bf16 {
//     type Output = f32;
//     #[inline(always)]
//     fn _pow(self, rhs: u16) -> Self::Output {
//         self.to_f32().powf(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u16, max: u16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f32>]();
//             let min = min.[<to_ f32>]();
//             let max = max.[<to_ f32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u16, b: u16) -> Self::Output {
//         self.to_f32() * a.to_f32() + b.to_f32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u16) -> Self::Output {
//         self.to_f32() + rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u16) -> Self::Output {
//         self.to_f32() - rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u16) -> Self::Output {
//         self.to_f32() * rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u16) -> Self::Output {
//         self.to_f32() % rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u16) -> Self::Output {
//         self.to_f32().max(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u16) -> Self::Output {
//         self.to_f32().min(rhs.to_f32())
//     }
// }
// impl NormalOut<u32> for bf16 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: u32) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u32, max: u32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u32, b: u32) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u32) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u32) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u32) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u32) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u32) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u32) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<u64> for bf16 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: u64) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u64, max: u64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u64, b: u64) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u64) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u64) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u64) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u64) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u64) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u64) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<bf16> for bf16 {
//     type Output = bf16;
//     #[inline(always)]
//     fn _pow(self, rhs: bf16) -> Self::Output {
//         self.to_bf16().powf(rhs.to_bf16())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bf16, max: bf16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ bf16>]();
//             let min = min.[<to_ bf16>]();
//             let max = max.[<to_ bf16>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bf16, b: bf16) -> Self::Output {
//         self.to_bf16() * a.to_bf16() + b.to_bf16()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bf16) -> Self::Output {
//         self.to_bf16() + rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bf16) -> Self::Output {
//         self.to_bf16() - rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bf16) -> Self::Output {
//         self.to_bf16() * rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bf16) -> Self::Output {
//         self.to_bf16() % rhs.to_bf16()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bf16) -> Self::Output {
//         self.to_bf16().max(rhs.to_bf16())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bf16) -> Self::Output {
//         self.to_bf16().min(rhs.to_bf16())
//     }
// }
// impl NormalOut<isize> for bf16 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: isize) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: isize, max: isize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: isize, b: isize) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: isize) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: isize) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: isize) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: isize) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: isize) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: isize) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<usize> for bf16 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: usize) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: usize, max: usize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: usize, b: usize) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: usize) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: usize) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: usize) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: usize) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: usize) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: usize) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<Complex32> for bf16 {
//     type Output = Complex32;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32().powc(rhs.to_complex32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex32, max: Complex32) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex32>]();
//             let min = min.[<to_ complex32>]();
//             let max = max.[<to_ complex32>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex32::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex32, b: Complex32) -> Self::Output {
//         self.to_complex32() * a.to_complex32() + b.to_complex32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() + rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() - rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() * rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() % rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex32) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex32) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<Complex64> for bf16 {
//     type Output = Complex64;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64().powc(rhs.to_complex64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex64, max: Complex64) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex64>]();
//             let min = min.[<to_ complex64>]();
//             let max = max.[<to_ complex64>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex64::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex64, b: Complex64) -> Self::Output {
//         self.to_complex64() * a.to_complex64() + b.to_complex64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() + rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() - rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() * rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() % rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex64) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex64) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<bool> for isize {
//     type Output = isize;
//     #[inline(always)]
//     fn _pow(self, rhs: bool) -> Self::Output {
//         self.to_isize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bool, max: bool) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ isize>]();
//             let min = min.[<to_ isize>]();
//             let max = max.[<to_ isize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bool, b: bool) -> Self::Output {
//         self.to_isize().wrapping_mul(a.to_isize()) + b.to_isize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bool) -> Self::Output {
//         self.to_isize().wrapping_add(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bool) -> Self::Output {
//         self.to_isize().wrapping_sub(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bool) -> Self::Output {
//         self.to_isize().wrapping_mul(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bool) -> Self::Output {
//         self.to_isize().wrapping_rem(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bool) -> Self::Output {
//         self.to_isize().max(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bool) -> Self::Output {
//         self.to_isize().min(rhs.to_isize())
//     }
// }
// impl NormalOut<half::f16> for isize {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: half::f16) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: half::f16, max: half::f16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: half::f16, b: half::f16) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: half::f16) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: half::f16) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: half::f16) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: half::f16) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: half::f16) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: half::f16) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<f32> for isize {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: f32) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f32, max: f32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f32, b: f32) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f32) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f32) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f32) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f32) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f32) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f32) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<f64> for isize {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: f64) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f64, max: f64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f64, b: f64) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f64) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f64) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f64) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f64) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f64) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f64) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<i8> for isize {
//     type Output = isize;
//     #[inline(always)]
//     fn _pow(self, rhs: i8) -> Self::Output {
//         self.to_isize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i8, max: i8) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ isize>]();
//             let min = min.[<to_ isize>]();
//             let max = max.[<to_ isize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i8, b: i8) -> Self::Output {
//         self.to_isize().wrapping_mul(a.to_isize()) + b.to_isize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i8) -> Self::Output {
//         self.to_isize().wrapping_add(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i8) -> Self::Output {
//         self.to_isize().wrapping_sub(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i8) -> Self::Output {
//         self.to_isize().wrapping_mul(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i8) -> Self::Output {
//         self.to_isize().wrapping_rem(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i8) -> Self::Output {
//         self.to_isize().max(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i8) -> Self::Output {
//         self.to_isize().min(rhs.to_isize())
//     }
// }
// impl NormalOut<i16> for isize {
//     type Output = isize;
//     #[inline(always)]
//     fn _pow(self, rhs: i16) -> Self::Output {
//         self.to_isize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i16, max: i16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ isize>]();
//             let min = min.[<to_ isize>]();
//             let max = max.[<to_ isize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i16, b: i16) -> Self::Output {
//         self.to_isize().wrapping_mul(a.to_isize()) + b.to_isize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i16) -> Self::Output {
//         self.to_isize().wrapping_add(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i16) -> Self::Output {
//         self.to_isize().wrapping_sub(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i16) -> Self::Output {
//         self.to_isize().wrapping_mul(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i16) -> Self::Output {
//         self.to_isize().wrapping_rem(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i16) -> Self::Output {
//         self.to_isize().max(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i16) -> Self::Output {
//         self.to_isize().min(rhs.to_isize())
//     }
// }
// impl NormalOut<i32> for isize {
//     type Output = isize;
//     #[inline(always)]
//     fn _pow(self, rhs: i32) -> Self::Output {
//         self.to_isize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i32, max: i32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ isize>]();
//             let min = min.[<to_ isize>]();
//             let max = max.[<to_ isize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i32, b: i32) -> Self::Output {
//         self.to_isize().wrapping_mul(a.to_isize()) + b.to_isize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i32) -> Self::Output {
//         self.to_isize().wrapping_add(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i32) -> Self::Output {
//         self.to_isize().wrapping_sub(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i32) -> Self::Output {
//         self.to_isize().wrapping_mul(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i32) -> Self::Output {
//         self.to_isize().wrapping_rem(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i32) -> Self::Output {
//         self.to_isize().max(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i32) -> Self::Output {
//         self.to_isize().min(rhs.to_isize())
//     }
// }
// impl NormalOut<i64> for isize {
//     type Output = isize;
//     #[inline(always)]
//     fn _pow(self, rhs: i64) -> Self::Output {
//         self.to_isize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i64, max: i64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ isize>]();
//             let min = min.[<to_ isize>]();
//             let max = max.[<to_ isize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i64, b: i64) -> Self::Output {
//         self.to_isize().wrapping_mul(a.to_isize()) + b.to_isize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i64) -> Self::Output {
//         self.to_isize().wrapping_add(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i64) -> Self::Output {
//         self.to_isize().wrapping_sub(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i64) -> Self::Output {
//         self.to_isize().wrapping_mul(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i64) -> Self::Output {
//         self.to_isize().wrapping_rem(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i64) -> Self::Output {
//         self.to_isize().max(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i64) -> Self::Output {
//         self.to_isize().min(rhs.to_isize())
//     }
// }
// impl NormalOut<u8> for isize {
//     type Output = isize;
//     #[inline(always)]
//     fn _pow(self, rhs: u8) -> Self::Output {
//         self.to_isize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u8, max: u8) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ isize>]();
//             let min = min.[<to_ isize>]();
//             let max = max.[<to_ isize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u8, b: u8) -> Self::Output {
//         self.to_isize().wrapping_mul(a.to_isize()) + b.to_isize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u8) -> Self::Output {
//         self.to_isize().wrapping_add(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u8) -> Self::Output {
//         self.to_isize().wrapping_sub(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u8) -> Self::Output {
//         self.to_isize().wrapping_mul(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u8) -> Self::Output {
//         self.to_isize().wrapping_rem(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u8) -> Self::Output {
//         self.to_isize().max(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u8) -> Self::Output {
//         self.to_isize().min(rhs.to_isize())
//     }
// }
// impl NormalOut<u16> for isize {
//     type Output = isize;
//     #[inline(always)]
//     fn _pow(self, rhs: u16) -> Self::Output {
//         self.to_isize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u16, max: u16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ isize>]();
//             let min = min.[<to_ isize>]();
//             let max = max.[<to_ isize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u16, b: u16) -> Self::Output {
//         self.to_isize().wrapping_mul(a.to_isize()) + b.to_isize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u16) -> Self::Output {
//         self.to_isize().wrapping_add(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u16) -> Self::Output {
//         self.to_isize().wrapping_sub(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u16) -> Self::Output {
//         self.to_isize().wrapping_mul(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u16) -> Self::Output {
//         self.to_isize().wrapping_rem(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u16) -> Self::Output {
//         self.to_isize().max(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u16) -> Self::Output {
//         self.to_isize().min(rhs.to_isize())
//     }
// }
// impl NormalOut<u32> for isize {
//     type Output = isize;
//     #[inline(always)]
//     fn _pow(self, rhs: u32) -> Self::Output {
//         self.to_isize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u32, max: u32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ isize>]();
//             let min = min.[<to_ isize>]();
//             let max = max.[<to_ isize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u32, b: u32) -> Self::Output {
//         self.to_isize().wrapping_mul(a.to_isize()) + b.to_isize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u32) -> Self::Output {
//         self.to_isize().wrapping_add(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u32) -> Self::Output {
//         self.to_isize().wrapping_sub(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u32) -> Self::Output {
//         self.to_isize().wrapping_mul(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u32) -> Self::Output {
//         self.to_isize().wrapping_rem(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u32) -> Self::Output {
//         self.to_isize().max(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u32) -> Self::Output {
//         self.to_isize().min(rhs.to_isize())
//     }
// }
// impl NormalOut<u64> for isize {
//     type Output = isize;
//     #[inline(always)]
//     fn _pow(self, rhs: u64) -> Self::Output {
//         self.to_isize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u64, max: u64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ isize>]();
//             let min = min.[<to_ isize>]();
//             let max = max.[<to_ isize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u64, b: u64) -> Self::Output {
//         self.to_isize().wrapping_mul(a.to_isize()) + b.to_isize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u64) -> Self::Output {
//         self.to_isize().wrapping_add(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u64) -> Self::Output {
//         self.to_isize().wrapping_sub(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u64) -> Self::Output {
//         self.to_isize().wrapping_mul(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u64) -> Self::Output {
//         self.to_isize().wrapping_rem(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u64) -> Self::Output {
//         self.to_isize().max(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u64) -> Self::Output {
//         self.to_isize().min(rhs.to_isize())
//     }
// }
// impl NormalOut<bf16> for isize {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: bf16) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bf16, max: bf16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bf16, b: bf16) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bf16) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bf16) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bf16) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bf16) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bf16) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bf16) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<isize> for isize {
//     type Output = isize;
//     #[inline(always)]
//     fn _pow(self, rhs: isize) -> Self::Output {
//         self.to_isize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: isize, max: isize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ isize>]();
//             let min = min.[<to_ isize>]();
//             let max = max.[<to_ isize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: isize, b: isize) -> Self::Output {
//         self.to_isize().wrapping_mul(a.to_isize()) + b.to_isize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_add(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_sub(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_mul(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_rem(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: isize) -> Self::Output {
//         self.to_isize().max(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: isize) -> Self::Output {
//         self.to_isize().min(rhs.to_isize())
//     }
// }
// impl NormalOut<usize> for isize {
//     type Output = isize;
//     #[inline(always)]
//     fn _pow(self, rhs: usize) -> Self::Output {
//         self.to_isize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: usize, max: usize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ isize>]();
//             let min = min.[<to_ isize>]();
//             let max = max.[<to_ isize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: usize, b: usize) -> Self::Output {
//         self.to_isize().wrapping_mul(a.to_isize()) + b.to_isize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: usize) -> Self::Output {
//         self.to_isize().wrapping_add(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: usize) -> Self::Output {
//         self.to_isize().wrapping_sub(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: usize) -> Self::Output {
//         self.to_isize().wrapping_mul(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: usize) -> Self::Output {
//         self.to_isize().wrapping_rem(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: usize) -> Self::Output {
//         self.to_isize().max(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: usize) -> Self::Output {
//         self.to_isize().min(rhs.to_isize())
//     }
// }
// impl NormalOut<Complex32> for isize {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex32) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex32, max: Complex32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex32, b: Complex32) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex32) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex32) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex32) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex32) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex32) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex32) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<Complex64> for isize {
//     type Output = Complex64;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64().powc(rhs.to_complex64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex64, max: Complex64) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex64>]();
//             let min = min.[<to_ complex64>]();
//             let max = max.[<to_ complex64>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex64::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex64, b: Complex64) -> Self::Output {
//         self.to_complex64() * a.to_complex64() + b.to_complex64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() + rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() - rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() * rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() % rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex64) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex64) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<bool> for usize {
//     type Output = usize;
//     #[inline(always)]
//     fn _pow(self, rhs: bool) -> Self::Output {
//         self.to_usize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bool, max: bool) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ usize>]();
//             let min = min.[<to_ usize>]();
//             let max = max.[<to_ usize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bool, b: bool) -> Self::Output {
//         self.to_usize().wrapping_mul(a.to_usize()) + b.to_usize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bool) -> Self::Output {
//         self.to_usize().wrapping_add(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bool) -> Self::Output {
//         self.to_usize().wrapping_sub(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bool) -> Self::Output {
//         self.to_usize().wrapping_mul(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bool) -> Self::Output {
//         self.to_usize().wrapping_rem(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bool) -> Self::Output {
//         self.to_usize().max(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bool) -> Self::Output {
//         self.to_usize().min(rhs.to_usize())
//     }
// }
// impl NormalOut<half::f16> for usize {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: half::f16) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: half::f16, max: half::f16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: half::f16, b: half::f16) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: half::f16) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: half::f16) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: half::f16) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: half::f16) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: half::f16) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: half::f16) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<f32> for usize {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: f32) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f32, max: f32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f32, b: f32) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f32) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f32) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f32) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f32) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f32) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f32) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<f64> for usize {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: f64) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f64, max: f64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f64, b: f64) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f64) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f64) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f64) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f64) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f64) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f64) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<i8> for usize {
//     type Output = usize;
//     #[inline(always)]
//     fn _pow(self, rhs: i8) -> Self::Output {
//         self.to_usize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i8, max: i8) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ usize>]();
//             let min = min.[<to_ usize>]();
//             let max = max.[<to_ usize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i8, b: i8) -> Self::Output {
//         self.to_usize().wrapping_mul(a.to_usize()) + b.to_usize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i8) -> Self::Output {
//         self.to_usize().wrapping_add(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i8) -> Self::Output {
//         self.to_usize().wrapping_sub(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i8) -> Self::Output {
//         self.to_usize().wrapping_mul(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i8) -> Self::Output {
//         self.to_usize().wrapping_rem(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i8) -> Self::Output {
//         self.to_usize().max(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i8) -> Self::Output {
//         self.to_usize().min(rhs.to_usize())
//     }
// }
// impl NormalOut<i16> for usize {
//     type Output = usize;
//     #[inline(always)]
//     fn _pow(self, rhs: i16) -> Self::Output {
//         self.to_usize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i16, max: i16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ usize>]();
//             let min = min.[<to_ usize>]();
//             let max = max.[<to_ usize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i16, b: i16) -> Self::Output {
//         self.to_usize().wrapping_mul(a.to_usize()) + b.to_usize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i16) -> Self::Output {
//         self.to_usize().wrapping_add(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i16) -> Self::Output {
//         self.to_usize().wrapping_sub(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i16) -> Self::Output {
//         self.to_usize().wrapping_mul(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i16) -> Self::Output {
//         self.to_usize().wrapping_rem(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i16) -> Self::Output {
//         self.to_usize().max(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i16) -> Self::Output {
//         self.to_usize().min(rhs.to_usize())
//     }
// }
// impl NormalOut<i32> for usize {
//     type Output = usize;
//     #[inline(always)]
//     fn _pow(self, rhs: i32) -> Self::Output {
//         self.to_usize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i32, max: i32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ usize>]();
//             let min = min.[<to_ usize>]();
//             let max = max.[<to_ usize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i32, b: i32) -> Self::Output {
//         self.to_usize().wrapping_mul(a.to_usize()) + b.to_usize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i32) -> Self::Output {
//         self.to_usize().wrapping_add(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i32) -> Self::Output {
//         self.to_usize().wrapping_sub(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i32) -> Self::Output {
//         self.to_usize().wrapping_mul(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i32) -> Self::Output {
//         self.to_usize().wrapping_rem(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i32) -> Self::Output {
//         self.to_usize().max(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i32) -> Self::Output {
//         self.to_usize().min(rhs.to_usize())
//     }
// }
// impl NormalOut<i64> for usize {
//     type Output = isize;
//     #[inline(always)]
//     fn _pow(self, rhs: i64) -> Self::Output {
//         self.to_isize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i64, max: i64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ isize>]();
//             let min = min.[<to_ isize>]();
//             let max = max.[<to_ isize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i64, b: i64) -> Self::Output {
//         self.to_isize().wrapping_mul(a.to_isize()) + b.to_isize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i64) -> Self::Output {
//         self.to_isize().wrapping_add(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i64) -> Self::Output {
//         self.to_isize().wrapping_sub(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i64) -> Self::Output {
//         self.to_isize().wrapping_mul(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i64) -> Self::Output {
//         self.to_isize().wrapping_rem(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i64) -> Self::Output {
//         self.to_isize().max(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i64) -> Self::Output {
//         self.to_isize().min(rhs.to_isize())
//     }
// }
// impl NormalOut<u8> for usize {
//     type Output = usize;
//     #[inline(always)]
//     fn _pow(self, rhs: u8) -> Self::Output {
//         self.to_usize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u8, max: u8) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ usize>]();
//             let min = min.[<to_ usize>]();
//             let max = max.[<to_ usize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u8, b: u8) -> Self::Output {
//         self.to_usize().wrapping_mul(a.to_usize()) + b.to_usize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u8) -> Self::Output {
//         self.to_usize().wrapping_add(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u8) -> Self::Output {
//         self.to_usize().wrapping_sub(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u8) -> Self::Output {
//         self.to_usize().wrapping_mul(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u8) -> Self::Output {
//         self.to_usize().wrapping_rem(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u8) -> Self::Output {
//         self.to_usize().max(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u8) -> Self::Output {
//         self.to_usize().min(rhs.to_usize())
//     }
// }
// impl NormalOut<u16> for usize {
//     type Output = usize;
//     #[inline(always)]
//     fn _pow(self, rhs: u16) -> Self::Output {
//         self.to_usize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u16, max: u16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ usize>]();
//             let min = min.[<to_ usize>]();
//             let max = max.[<to_ usize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u16, b: u16) -> Self::Output {
//         self.to_usize().wrapping_mul(a.to_usize()) + b.to_usize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u16) -> Self::Output {
//         self.to_usize().wrapping_add(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u16) -> Self::Output {
//         self.to_usize().wrapping_sub(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u16) -> Self::Output {
//         self.to_usize().wrapping_mul(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u16) -> Self::Output {
//         self.to_usize().wrapping_rem(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u16) -> Self::Output {
//         self.to_usize().max(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u16) -> Self::Output {
//         self.to_usize().min(rhs.to_usize())
//     }
// }
// impl NormalOut<u32> for usize {
//     type Output = usize;
//     #[inline(always)]
//     fn _pow(self, rhs: u32) -> Self::Output {
//         self.to_usize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u32, max: u32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ usize>]();
//             let min = min.[<to_ usize>]();
//             let max = max.[<to_ usize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u32, b: u32) -> Self::Output {
//         self.to_usize().wrapping_mul(a.to_usize()) + b.to_usize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u32) -> Self::Output {
//         self.to_usize().wrapping_add(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u32) -> Self::Output {
//         self.to_usize().wrapping_sub(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u32) -> Self::Output {
//         self.to_usize().wrapping_mul(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u32) -> Self::Output {
//         self.to_usize().wrapping_rem(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u32) -> Self::Output {
//         self.to_usize().max(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u32) -> Self::Output {
//         self.to_usize().min(rhs.to_usize())
//     }
// }
// impl NormalOut<u64> for usize {
//     type Output = usize;
//     #[inline(always)]
//     fn _pow(self, rhs: u64) -> Self::Output {
//         self.to_usize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u64, max: u64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ usize>]();
//             let min = min.[<to_ usize>]();
//             let max = max.[<to_ usize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u64, b: u64) -> Self::Output {
//         self.to_usize().wrapping_mul(a.to_usize()) + b.to_usize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u64) -> Self::Output {
//         self.to_usize().wrapping_add(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u64) -> Self::Output {
//         self.to_usize().wrapping_sub(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u64) -> Self::Output {
//         self.to_usize().wrapping_mul(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u64) -> Self::Output {
//         self.to_usize().wrapping_rem(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u64) -> Self::Output {
//         self.to_usize().max(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u64) -> Self::Output {
//         self.to_usize().min(rhs.to_usize())
//     }
// }
// impl NormalOut<bf16> for usize {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: bf16) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bf16, max: bf16) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bf16, b: bf16) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bf16) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bf16) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bf16) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bf16) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bf16) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bf16) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<isize> for usize {
//     type Output = isize;
//     #[inline(always)]
//     fn _pow(self, rhs: isize) -> Self::Output {
//         self.to_isize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: isize, max: isize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ isize>]();
//             let min = min.[<to_ isize>]();
//             let max = max.[<to_ isize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: isize, b: isize) -> Self::Output {
//         self.to_isize().wrapping_mul(a.to_isize()) + b.to_isize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_add(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_sub(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_mul(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: isize) -> Self::Output {
//         self.to_isize().wrapping_rem(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: isize) -> Self::Output {
//         self.to_isize().max(rhs.to_isize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: isize) -> Self::Output {
//         self.to_isize().min(rhs.to_isize())
//     }
// }
// impl NormalOut<usize> for usize {
//     type Output = usize;
//     #[inline(always)]
//     fn _pow(self, rhs: usize) -> Self::Output {
//         self.to_usize().pow(rhs.to_u32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: usize, max: usize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ usize>]();
//             let min = min.[<to_ usize>]();
//             let max = max.[<to_ usize>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: usize, b: usize) -> Self::Output {
//         self.to_usize().wrapping_mul(a.to_usize()) + b.to_usize()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: usize) -> Self::Output {
//         self.to_usize().wrapping_add(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: usize) -> Self::Output {
//         self.to_usize().wrapping_sub(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: usize) -> Self::Output {
//         self.to_usize().wrapping_mul(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: usize) -> Self::Output {
//         self.to_usize().wrapping_rem(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _max(self, rhs: usize) -> Self::Output {
//         self.to_usize().max(rhs.to_usize())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: usize) -> Self::Output {
//         self.to_usize().min(rhs.to_usize())
//     }
// }
// impl NormalOut<Complex32> for usize {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex32) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex32, max: Complex32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex32, b: Complex32) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex32) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex32) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex32) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex32) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex32) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex32) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<Complex64> for usize {
//     type Output = Complex64;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64().powc(rhs.to_complex64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex64, max: Complex64) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex64>]();
//             let min = min.[<to_ complex64>]();
//             let max = max.[<to_ complex64>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex64::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex64, b: Complex64) -> Self::Output {
//         self.to_complex64() * a.to_complex64() + b.to_complex64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() + rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() - rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() * rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() % rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex64) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex64) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<bool> for Complex32 {
//     type Output = Complex32;
//     #[inline(always)]
//     fn _pow(self, rhs: bool) -> Self::Output {
//         self.to_complex32().powc(rhs.to_complex32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bool, max: bool) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex32>]();
//             let min = min.[<to_ complex32>]();
//             let max = max.[<to_ complex32>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex32::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bool, b: bool) -> Self::Output {
//         self.to_complex32() * a.to_complex32() + b.to_complex32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bool) -> Self::Output {
//         self.to_complex32() + rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bool) -> Self::Output {
//         self.to_complex32() - rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bool) -> Self::Output {
//         self.to_complex32() * rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bool) -> Self::Output {
//         self.to_complex32() % rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bool) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bool) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<half::f16> for Complex32 {
//     type Output = Complex32;
//     #[inline(always)]
//     fn _pow(self, rhs: half::f16) -> Self::Output {
//         self.to_complex32().powc(rhs.to_complex32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: half::f16, max: half::f16) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex32>]();
//             let min = min.[<to_ complex32>]();
//             let max = max.[<to_ complex32>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex32::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: half::f16, b: half::f16) -> Self::Output {
//         self.to_complex32() * a.to_complex32() + b.to_complex32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: half::f16) -> Self::Output {
//         self.to_complex32() + rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: half::f16) -> Self::Output {
//         self.to_complex32() - rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: half::f16) -> Self::Output {
//         self.to_complex32() * rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: half::f16) -> Self::Output {
//         self.to_complex32() % rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: half::f16) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: half::f16) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<f32> for Complex32 {
//     type Output = f32;
//     #[inline(always)]
//     fn _pow(self, rhs: f32) -> Self::Output {
//         self.to_f32().powf(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f32, max: f32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f32>]();
//             let min = min.[<to_ f32>]();
//             let max = max.[<to_ f32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f32, b: f32) -> Self::Output {
//         self.to_f32() * a.to_f32() + b.to_f32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f32) -> Self::Output {
//         self.to_f32() + rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f32) -> Self::Output {
//         self.to_f32() - rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f32) -> Self::Output {
//         self.to_f32() * rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f32) -> Self::Output {
//         self.to_f32() % rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f32) -> Self::Output {
//         self.to_f32().max(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f32) -> Self::Output {
//         self.to_f32().min(rhs.to_f32())
//     }
// }
// impl NormalOut<f64> for Complex32 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: f64) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f64, max: f64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f64, b: f64) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f64) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f64) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f64) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f64) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f64) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f64) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<i8> for Complex32 {
//     type Output = Complex32;
//     #[inline(always)]
//     fn _pow(self, rhs: i8) -> Self::Output {
//         self.to_complex32().powc(rhs.to_complex32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i8, max: i8) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex32>]();
//             let min = min.[<to_ complex32>]();
//             let max = max.[<to_ complex32>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex32::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i8, b: i8) -> Self::Output {
//         self.to_complex32() * a.to_complex32() + b.to_complex32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i8) -> Self::Output {
//         self.to_complex32() + rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i8) -> Self::Output {
//         self.to_complex32() - rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i8) -> Self::Output {
//         self.to_complex32() * rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i8) -> Self::Output {
//         self.to_complex32() % rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i8) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i8) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<i16> for Complex32 {
//     type Output = Complex32;
//     #[inline(always)]
//     fn _pow(self, rhs: i16) -> Self::Output {
//         self.to_complex32().powc(rhs.to_complex32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i16, max: i16) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex32>]();
//             let min = min.[<to_ complex32>]();
//             let max = max.[<to_ complex32>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex32::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i16, b: i16) -> Self::Output {
//         self.to_complex32() * a.to_complex32() + b.to_complex32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i16) -> Self::Output {
//         self.to_complex32() + rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i16) -> Self::Output {
//         self.to_complex32() - rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i16) -> Self::Output {
//         self.to_complex32() * rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i16) -> Self::Output {
//         self.to_complex32() % rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i16) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i16) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<i32> for Complex32 {
//     type Output = f32;
//     #[inline(always)]
//     fn _pow(self, rhs: i32) -> Self::Output {
//         self.to_f32().powf(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i32, max: i32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f32>]();
//             let min = min.[<to_ f32>]();
//             let max = max.[<to_ f32>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i32, b: i32) -> Self::Output {
//         self.to_f32() * a.to_f32() + b.to_f32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i32) -> Self::Output {
//         self.to_f32() + rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i32) -> Self::Output {
//         self.to_f32() - rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i32) -> Self::Output {
//         self.to_f32() * rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i32) -> Self::Output {
//         self.to_f32() % rhs.to_f32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i32) -> Self::Output {
//         self.to_f32().max(rhs.to_f32())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i32) -> Self::Output {
//         self.to_f32().min(rhs.to_f32())
//     }
// }
// impl NormalOut<i64> for Complex32 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: i64) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i64, max: i64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i64, b: i64) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i64) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i64) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i64) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i64) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i64) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i64) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<u8> for Complex32 {
//     type Output = Complex32;
//     #[inline(always)]
//     fn _pow(self, rhs: u8) -> Self::Output {
//         self.to_complex32().powc(rhs.to_complex32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u8, max: u8) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex32>]();
//             let min = min.[<to_ complex32>]();
//             let max = max.[<to_ complex32>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex32::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u8, b: u8) -> Self::Output {
//         self.to_complex32() * a.to_complex32() + b.to_complex32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u8) -> Self::Output {
//         self.to_complex32() + rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u8) -> Self::Output {
//         self.to_complex32() - rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u8) -> Self::Output {
//         self.to_complex32() * rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u8) -> Self::Output {
//         self.to_complex32() % rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u8) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u8) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<u16> for Complex32 {
//     type Output = Complex32;
//     #[inline(always)]
//     fn _pow(self, rhs: u16) -> Self::Output {
//         self.to_complex32().powc(rhs.to_complex32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u16, max: u16) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex32>]();
//             let min = min.[<to_ complex32>]();
//             let max = max.[<to_ complex32>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex32::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u16, b: u16) -> Self::Output {
//         self.to_complex32() * a.to_complex32() + b.to_complex32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u16) -> Self::Output {
//         self.to_complex32() + rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u16) -> Self::Output {
//         self.to_complex32() - rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u16) -> Self::Output {
//         self.to_complex32() * rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u16) -> Self::Output {
//         self.to_complex32() % rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u16) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u16) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<u32> for Complex32 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: u32) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u32, max: u32) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u32, b: u32) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u32) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u32) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u32) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u32) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u32) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u32) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<u64> for Complex32 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: u64) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u64, max: u64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u64, b: u64) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u64) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u64) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u64) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u64) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u64) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u64) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<bf16> for Complex32 {
//     type Output = Complex32;
//     #[inline(always)]
//     fn _pow(self, rhs: bf16) -> Self::Output {
//         self.to_complex32().powc(rhs.to_complex32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bf16, max: bf16) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex32>]();
//             let min = min.[<to_ complex32>]();
//             let max = max.[<to_ complex32>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex32::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bf16, b: bf16) -> Self::Output {
//         self.to_complex32() * a.to_complex32() + b.to_complex32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bf16) -> Self::Output {
//         self.to_complex32() + rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bf16) -> Self::Output {
//         self.to_complex32() - rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bf16) -> Self::Output {
//         self.to_complex32() * rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bf16) -> Self::Output {
//         self.to_complex32() % rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bf16) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bf16) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<isize> for Complex32 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: isize) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: isize, max: isize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: isize, b: isize) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: isize) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: isize) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: isize) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: isize) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: isize) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: isize) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<usize> for Complex32 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: usize) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: usize, max: usize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: usize, b: usize) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: usize) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: usize) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: usize) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: usize) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: usize) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: usize) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<Complex32> for Complex32 {
//     type Output = Complex32;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32().powc(rhs.to_complex32())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex32, max: Complex32) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex32>]();
//             let min = min.[<to_ complex32>]();
//             let max = max.[<to_ complex32>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex32::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex32, b: Complex32) -> Self::Output {
//         self.to_complex32() * a.to_complex32() + b.to_complex32()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() + rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() - rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() * rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex32) -> Self::Output {
//         self.to_complex32() % rhs.to_complex32()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex32) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex32) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<Complex64> for Complex32 {
//     type Output = Complex64;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64().powc(rhs.to_complex64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex64, max: Complex64) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex64>]();
//             let min = min.[<to_ complex64>]();
//             let max = max.[<to_ complex64>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex64::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex64, b: Complex64) -> Self::Output {
//         self.to_complex64() * a.to_complex64() + b.to_complex64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() + rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() - rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() * rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() % rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex64) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex64) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<bool> for Complex64 {
//     type Output = Complex64;
//     #[inline(always)]
//     fn _pow(self, rhs: bool) -> Self::Output {
//         self.to_complex64().powc(rhs.to_complex64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bool, max: bool) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex64>]();
//             let min = min.[<to_ complex64>]();
//             let max = max.[<to_ complex64>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex64::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bool, b: bool) -> Self::Output {
//         self.to_complex64() * a.to_complex64() + b.to_complex64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bool) -> Self::Output {
//         self.to_complex64() + rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bool) -> Self::Output {
//         self.to_complex64() - rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bool) -> Self::Output {
//         self.to_complex64() * rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bool) -> Self::Output {
//         self.to_complex64() % rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bool) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bool) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<half::f16> for Complex64 {
//     type Output = Complex64;
//     #[inline(always)]
//     fn _pow(self, rhs: half::f16) -> Self::Output {
//         self.to_complex64().powc(rhs.to_complex64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: half::f16, max: half::f16) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex64>]();
//             let min = min.[<to_ complex64>]();
//             let max = max.[<to_ complex64>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex64::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: half::f16, b: half::f16) -> Self::Output {
//         self.to_complex64() * a.to_complex64() + b.to_complex64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: half::f16) -> Self::Output {
//         self.to_complex64() + rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: half::f16) -> Self::Output {
//         self.to_complex64() - rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: half::f16) -> Self::Output {
//         self.to_complex64() * rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: half::f16) -> Self::Output {
//         self.to_complex64() % rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: half::f16) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: half::f16) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<f32> for Complex64 {
//     type Output = Complex64;
//     #[inline(always)]
//     fn _pow(self, rhs: f32) -> Self::Output {
//         self.to_complex64().powc(rhs.to_complex64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f32, max: f32) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex64>]();
//             let min = min.[<to_ complex64>]();
//             let max = max.[<to_ complex64>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex64::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f32, b: f32) -> Self::Output {
//         self.to_complex64() * a.to_complex64() + b.to_complex64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f32) -> Self::Output {
//         self.to_complex64() + rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f32) -> Self::Output {
//         self.to_complex64() - rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f32) -> Self::Output {
//         self.to_complex64() * rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f32) -> Self::Output {
//         self.to_complex64() % rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f32) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f32) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<f64> for Complex64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: f64) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: f64, max: f64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: f64, b: f64) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: f64) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: f64) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: f64) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: f64) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: f64) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: f64) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<i8> for Complex64 {
//     type Output = Complex64;
//     #[inline(always)]
//     fn _pow(self, rhs: i8) -> Self::Output {
//         self.to_complex64().powc(rhs.to_complex64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i8, max: i8) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex64>]();
//             let min = min.[<to_ complex64>]();
//             let max = max.[<to_ complex64>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex64::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i8, b: i8) -> Self::Output {
//         self.to_complex64() * a.to_complex64() + b.to_complex64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i8) -> Self::Output {
//         self.to_complex64() + rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i8) -> Self::Output {
//         self.to_complex64() - rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i8) -> Self::Output {
//         self.to_complex64() * rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i8) -> Self::Output {
//         self.to_complex64() % rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i8) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i8) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<i16> for Complex64 {
//     type Output = Complex64;
//     #[inline(always)]
//     fn _pow(self, rhs: i16) -> Self::Output {
//         self.to_complex64().powc(rhs.to_complex64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i16, max: i16) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex64>]();
//             let min = min.[<to_ complex64>]();
//             let max = max.[<to_ complex64>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex64::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i16, b: i16) -> Self::Output {
//         self.to_complex64() * a.to_complex64() + b.to_complex64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i16) -> Self::Output {
//         self.to_complex64() + rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i16) -> Self::Output {
//         self.to_complex64() - rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i16) -> Self::Output {
//         self.to_complex64() * rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i16) -> Self::Output {
//         self.to_complex64() % rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i16) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i16) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<i32> for Complex64 {
//     type Output = Complex64;
//     #[inline(always)]
//     fn _pow(self, rhs: i32) -> Self::Output {
//         self.to_complex64().powc(rhs.to_complex64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i32, max: i32) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex64>]();
//             let min = min.[<to_ complex64>]();
//             let max = max.[<to_ complex64>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex64::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i32, b: i32) -> Self::Output {
//         self.to_complex64() * a.to_complex64() + b.to_complex64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i32) -> Self::Output {
//         self.to_complex64() + rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i32) -> Self::Output {
//         self.to_complex64() - rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i32) -> Self::Output {
//         self.to_complex64() * rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i32) -> Self::Output {
//         self.to_complex64() % rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i32) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i32) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<i64> for Complex64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: i64) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: i64, max: i64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: i64, b: i64) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: i64) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: i64) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: i64) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: i64) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: i64) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: i64) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<u8> for Complex64 {
//     type Output = Complex64;
//     #[inline(always)]
//     fn _pow(self, rhs: u8) -> Self::Output {
//         self.to_complex64().powc(rhs.to_complex64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u8, max: u8) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex64>]();
//             let min = min.[<to_ complex64>]();
//             let max = max.[<to_ complex64>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex64::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u8, b: u8) -> Self::Output {
//         self.to_complex64() * a.to_complex64() + b.to_complex64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u8) -> Self::Output {
//         self.to_complex64() + rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u8) -> Self::Output {
//         self.to_complex64() - rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u8) -> Self::Output {
//         self.to_complex64() * rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u8) -> Self::Output {
//         self.to_complex64() % rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u8) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u8) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<u16> for Complex64 {
//     type Output = Complex64;
//     #[inline(always)]
//     fn _pow(self, rhs: u16) -> Self::Output {
//         self.to_complex64().powc(rhs.to_complex64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u16, max: u16) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex64>]();
//             let min = min.[<to_ complex64>]();
//             let max = max.[<to_ complex64>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex64::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u16, b: u16) -> Self::Output {
//         self.to_complex64() * a.to_complex64() + b.to_complex64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u16) -> Self::Output {
//         self.to_complex64() + rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u16) -> Self::Output {
//         self.to_complex64() - rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u16) -> Self::Output {
//         self.to_complex64() * rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u16) -> Self::Output {
//         self.to_complex64() % rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u16) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u16) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<u32> for Complex64 {
//     type Output = Complex64;
//     #[inline(always)]
//     fn _pow(self, rhs: u32) -> Self::Output {
//         self.to_complex64().powc(rhs.to_complex64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u32, max: u32) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex64>]();
//             let min = min.[<to_ complex64>]();
//             let max = max.[<to_ complex64>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex64::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u32, b: u32) -> Self::Output {
//         self.to_complex64() * a.to_complex64() + b.to_complex64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u32) -> Self::Output {
//         self.to_complex64() + rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u32) -> Self::Output {
//         self.to_complex64() - rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u32) -> Self::Output {
//         self.to_complex64() * rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u32) -> Self::Output {
//         self.to_complex64() % rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u32) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u32) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<u64> for Complex64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: u64) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: u64, max: u64) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: u64, b: u64) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: u64) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: u64) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: u64) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: u64) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: u64) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: u64) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<bf16> for Complex64 {
//     type Output = Complex64;
//     #[inline(always)]
//     fn _pow(self, rhs: bf16) -> Self::Output {
//         self.to_complex64().powc(rhs.to_complex64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: bf16, max: bf16) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex64>]();
//             let min = min.[<to_ complex64>]();
//             let max = max.[<to_ complex64>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex64::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: bf16, b: bf16) -> Self::Output {
//         self.to_complex64() * a.to_complex64() + b.to_complex64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: bf16) -> Self::Output {
//         self.to_complex64() + rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: bf16) -> Self::Output {
//         self.to_complex64() - rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: bf16) -> Self::Output {
//         self.to_complex64() * rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: bf16) -> Self::Output {
//         self.to_complex64() % rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: bf16) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: bf16) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<isize> for Complex64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: isize) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: isize, max: isize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: isize, b: isize) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: isize) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: isize) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: isize) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: isize) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: isize) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: isize) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<usize> for Complex64 {
//     type Output = f64;
//     #[inline(always)]
//     fn _pow(self, rhs: usize) -> Self::Output {
//         self.to_f64().powf(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: usize, max: usize) -> Self::Output {
//         paste::paste! {
//             let a = self.[<to_ f64>]();
//             let min = min.[<to_ f64>]();
//             let max = max.[<to_ f64>]();
//             if a<min {
//                 min
//             }else if a>max {
//                 max
//             }else {
//                 a
//             }
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: usize, b: usize) -> Self::Output {
//         self.to_f64() * a.to_f64() + b.to_f64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: usize) -> Self::Output {
//         self.to_f64() + rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: usize) -> Self::Output {
//         self.to_f64() - rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: usize) -> Self::Output {
//         self.to_f64() * rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: usize) -> Self::Output {
//         self.to_f64() % rhs.to_f64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: usize) -> Self::Output {
//         self.to_f64().max(rhs.to_f64())
//     }
//     #[inline(always)]
//     fn _min(self, rhs: usize) -> Self::Output {
//         self.to_f64().min(rhs.to_f64())
//     }
// }
// impl NormalOut<Complex32> for Complex64 {
//     type Output = Complex64;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex32) -> Self::Output {
//         self.to_complex64().powc(rhs.to_complex64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex32, max: Complex32) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex64>]();
//             let min = min.[<to_ complex64>]();
//             let max = max.[<to_ complex64>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex64::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex32, b: Complex32) -> Self::Output {
//         self.to_complex64() * a.to_complex64() + b.to_complex64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex32) -> Self::Output {
//         self.to_complex64() + rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex32) -> Self::Output {
//         self.to_complex64() - rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex32) -> Self::Output {
//         self.to_complex64() * rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex32) -> Self::Output {
//         self.to_complex64() % rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex32) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex32) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
// impl NormalOut<Complex64> for Complex64 {
//     type Output = Complex64;
//     #[inline(always)]
//     fn _pow(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64().powc(rhs.to_complex64())
//     }
//     #[inline(always)]
//     fn _clip(self, min: Complex64, max: Complex64) -> Self::Output {
//         paste::paste! {
//             let c = self.[<to_ complex64>]();
//             let min = min.[<to_ complex64>]();
//             let max = max.[<to_ complex64>]();
//             let clamped_re = c.re.clamp(min.re,max.re);
//             let clamped_im = c.im.clamp(min.im,max.im);
//             Complex64::new(clamped_re,clamped_im)
//         }
//     }
//     #[inline(always)]
//     fn _mul_add(self, a: Complex64, b: Complex64) -> Self::Output {
//         self.to_complex64() * a.to_complex64() + b.to_complex64()
//     }
//     #[inline(always)]
//     fn _add(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() + rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _sub(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() - rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _mul(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() * rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _rem(self, rhs: Complex64) -> Self::Output {
//         self.to_complex64() % rhs.to_complex64()
//     }
//     #[inline(always)]
//     fn _max(self, rhs: Complex64) -> Self::Output {
//         panic!("max method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _min(self, rhs: Complex64) -> Self::Output {
//         panic!("min method is not supported for complex number")
//     }
// }
