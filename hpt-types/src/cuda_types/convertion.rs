use super::scalar::Scalar;
use half::bf16;
use half::f16;
use num_complex::Complex32;
use num_complex::Complex64;

/// The trait for converting cuda scalar
pub trait CudaConvertor {
    /// convert the value to bool
    fn to_bool(&self) -> Scalar<bool>;
    /// convert the value to u8
    fn to_u8(&self) -> Scalar<u8>;
    /// convert the value to u16
    fn to_u16(&self) -> Scalar<u16>;
    /// convert the value to u32
    fn to_u32(&self) -> Scalar<u32>;
    /// convert the value to u64
    fn to_u64(&self) -> Scalar<u64>;
    /// convert the value to usize
    fn to_usize(&self) -> Scalar<usize>;
    /// convert the value to i8
    fn to_i8(&self) -> Scalar<i8>;
    /// convert the value to i16
    fn to_i16(&self) -> Scalar<i16>;
    /// convert the value to i32
    fn to_i32(&self) -> Scalar<i32>;
    /// convert the value to i64
    fn to_i64(&self) -> Scalar<i64>;
    /// convert the value to isize
    fn to_isize(&self) -> Scalar<isize>;
    /// convert the value to f32
    fn to_f32(&self) -> Scalar<f32>;
    /// convert the value to f64
    fn to_f64(&self) -> Scalar<f64>;
    /// convert the value to f16
    fn to_f16(&self) -> Scalar<f16>;
    /// convert the value to bf16
    fn to_bf16(&self) -> Scalar<bf16>;
    /// convert the value to complex32
    fn to_complex32(&self) -> Scalar<Complex32>;
    /// convert the value to complex64
    fn to_complex64(&self) -> Scalar<Complex64>;
}

impl CudaConvertor for Scalar<bool> {
    #[inline(always)]
    fn to_bool(&self) -> Scalar<bool> {
        self.clone()
    }
    #[inline(always)]
    fn to_f16(&self) -> Scalar<half::f16> {
        Scalar::<half::f16>::new(format!("__uint2half((unsigned int){})", self.val()))
    }
    #[inline(always)]
    fn to_f32(&self) -> Scalar<f32> {
        Scalar::<f32>::new(format!("((float){})", self.val()))
    }
    #[inline(always)]
    fn to_f64(&self) -> Scalar<f64> {
        Scalar::<f64>::new(format!("((double){})", self.val()))
    }
    #[inline(always)]
    fn to_i8(&self) -> Scalar<i8> {
        Scalar::<i8>::new(format!("((char){})", self.val()))
    }
    #[inline(always)]
    fn to_i16(&self) -> Scalar<i16> {
        Scalar::<i16>::new(format!("((short){})", self.val()))
    }
    #[inline(always)]
    fn to_i32(&self) -> Scalar<i32> {
        Scalar::<i32>::new(format!("((int){})", self.val()))
    }
    #[inline(always)]
    fn to_i64(&self) -> Scalar<i64> {
        Scalar::<i64>::new(format!("((long long){})", self.val()))
    }
    #[inline(always)]
    fn to_u8(&self) -> Scalar<u8> {
        Scalar::<u8>::new(format!("((unsigned char){})", self.val()))
    }
    #[inline(always)]
    fn to_u16(&self) -> Scalar<u16> {
        Scalar::<u16>::new(format!("((unsigned short){})", self.val()))
    }
    #[inline(always)]
    fn to_u32(&self) -> Scalar<u32> {
        Scalar::<u32>::new(format!("((unsigned int){})", self.val()))
    }
    #[inline(always)]
    fn to_u64(&self) -> Scalar<u64> {
        Scalar::<u64>::new(format!("((unsigned long long){})", self.val()))
    }
    #[inline(always)]
    fn to_bf16(&self) -> Scalar<half::bf16> {
        Scalar::<half::bf16>::new(format!("__uint2bfloat16((unsigned int){})", self.val()))
    }
    #[inline(always)]
    fn to_isize(&self) -> Scalar<isize> {
        #[cfg(target_pointer_width = "64")]
        return Scalar::<isize>::new(format!("((long long){})", self.val()));
        #[cfg(target_pointer_width = "32")]
        return Scalar::<isize>::new(format!("((int){})", self.val()));
    }
    #[inline(always)]
    fn to_usize(&self) -> Scalar<usize> {
        #[cfg(target_pointer_width = "64")]
        return Scalar::<usize>::new(format!("((unsigned long long){})", self.val()));
        #[cfg(target_pointer_width = "32")]
        return Scalar::<usize>::new(format!("((unsigned int){})", self.val()));
    }
    #[inline(always)]
    fn to_complex32(&self) -> Scalar<Complex32> {
        Scalar::<Complex32>::new(format!("make_cuFloatComplex((float){}, 0.0f)", self.val()))
    }
    #[inline(always)]
    fn to_complex64(&self) -> Scalar<Complex64> {
        Scalar::<Complex64>::new(format!("make_cuDoubleComplex((double){}, 0.0)", self.val()))
    }
}
impl CudaConvertor for Scalar<f16> {
    #[inline(always)]
    fn to_bool(&self) -> Scalar<bool> {
        Scalar::new(format!("((bool)__half2int_rn({}))", self.val))
    }
    #[inline(always)]
    fn to_f16(&self) -> Scalar<half::f16> {
        self.clone()
    }
    #[inline(always)]
    fn to_f32(&self) -> Scalar<f32> {
        Scalar::new(format!("__half2float({})", self.val))
    }
    #[inline(always)]
    fn to_f64(&self) -> Scalar<f64> {
        Scalar::new(format!("(double)__half2float({})", self.val))
    }
    #[inline(always)]
    fn to_i8(&self) -> Scalar<i8> {
        Scalar::new(format!("((char)__half2int_rn({}))", self.val))
    }
    #[inline(always)]
    fn to_i16(&self) -> Scalar<i16> {
        Scalar::new(format!("((short)__half2int_rn({}))", self.val))
    }
    #[inline(always)]
    fn to_i32(&self) -> Scalar<i32> {
        Scalar::new(format!("__half2int_rn({})", self.val))
    }
    #[inline(always)]
    fn to_i64(&self) -> Scalar<i64> {
        Scalar::new(format!("__half2ll_rn({})", self.val))
    }
    #[inline(always)]
    fn to_u8(&self) -> Scalar<u8> {
        Scalar::new(format!("((unsigned char)__half2uint_rn({}))", self.val))
    }
    #[inline(always)]
    fn to_u16(&self) -> Scalar<u16> {
        Scalar::new(format!("((unsigned short)__half2uint_rn({}))", self.val))
    }
    #[inline(always)]
    fn to_u32(&self) -> Scalar<u32> {
        Scalar::new(format!("__half2uint_rn({})", self.val))
    }
    #[inline(always)]
    fn to_u64(&self) -> Scalar<u64> {
        Scalar::new(format!("__half2ull_rn({})", self.val))
    }
    #[inline(always)]
    fn to_bf16(&self) -> Scalar<half::bf16> {
        Scalar::new(format!("__float2bfloat16_rn(__half2float({}))", self.val))
    }
    #[inline(always)]
    fn to_isize(&self) -> Scalar<isize> {
        #[cfg(target_pointer_width = "64")]
        return Scalar::new(format!("__half2ll_rn({})", self.val));
        #[cfg(target_pointer_width = "32")]
        return Scalar::new(format!("__half2int_rn({})", self.val));
    }
    #[inline(always)]
    fn to_usize(&self) -> Scalar<usize> {
        #[cfg(target_pointer_width = "64")]
        return Scalar::new(format!("__half2ull_rn({})", self.val));
        #[cfg(target_pointer_width = "32")]
        return Scalar::new(format!("__half2uint_rn({})", self.val));
    }
    #[inline(always)]
    fn to_complex32(&self) -> Scalar<Complex32> {
        Scalar::new(format!(
            "make_cuFloatComplex(__half2float({}), 0.0f)",
            self.val
        ))
    }
    #[inline(always)]
    fn to_complex64(&self) -> Scalar<Complex64> {
        Scalar::new(format!(
            "make_cuDoubleComplex((double)__half2float({}), 0.0)",
            self.val
        ))
    }
}
impl CudaConvertor for Scalar<f32> {
    #[inline(always)]
    fn to_bool(&self) -> Scalar<bool> {
        Scalar::new(format!("((bool){})", self.val))
    }
    #[inline(always)]
    fn to_f16(&self) -> Scalar<f16> {
        Scalar::new(format!("__float2half_rn({})", self.val))
    }
    #[inline(always)]
    fn to_f32(&self) -> Scalar<f32> {
        self.clone()
    }
    #[inline(always)]
    fn to_f64(&self) -> Scalar<f64> {
        Scalar::new(format!("((double){})", self.val))
    }
    #[inline(always)]
    fn to_i8(&self) -> Scalar<i8> {
        Scalar::new(format!("((char){})", self.val))
    }
    #[inline(always)]
    fn to_i16(&self) -> Scalar<i16> {
        Scalar::new(format!("((short){})", self.val))
    }
    #[inline(always)]
    fn to_i32(&self) -> Scalar<i32> {
        Scalar::new(format!("((int){})", self.val))
    }
    #[inline(always)]
    fn to_i64(&self) -> Scalar<i64> {
        Scalar::new(format!("((long long){})", self.val))
    }
    #[inline(always)]
    fn to_u8(&self) -> Scalar<u8> {
        Scalar::new(format!("((unsigned char){})", self.val))
    }
    #[inline(always)]
    fn to_u16(&self) -> Scalar<u16> {
        Scalar::new(format!("((unsigned short){})", self.val))
    }
    #[inline(always)]
    fn to_u32(&self) -> Scalar<u32> {
        Scalar::new(format!("((unsigned int){})", self.val))
    }
    #[inline(always)]
    fn to_u64(&self) -> Scalar<u64> {
        Scalar::new(format!("((unsigned long long){})", self.val))
    }
    #[inline(always)]
    fn to_bf16(&self) -> Scalar<half::bf16> {
        Scalar::new(format!("__float2bfloat16_rn({})", self.val))
    }
    #[inline(always)]
    fn to_isize(&self) -> Scalar<isize> {
        #[cfg(target_pointer_width = "64")]
        return Scalar::new(format!("((long long){})", self.val));
        #[cfg(target_pointer_width = "32")]
        return Scalar::new(format!("((int){})", self.val));
    }
    #[inline(always)]
    fn to_usize(&self) -> Scalar<usize> {
        Scalar::new(format!("((size_t){})", self.val))
    }
    #[inline(always)]
    fn to_complex32(&self) -> Scalar<Complex32> {
        Scalar::new(format!("make_cuFloatComplex({}, 0.0f)", self.val))
    }
    #[inline(always)]
    fn to_complex64(&self) -> Scalar<Complex64> {
        Scalar::new(format!("make_cuDoubleComplex((double){}, 0.0)", self.val))
    }
}
impl CudaConvertor for Scalar<f64> {
    #[inline(always)]
    fn to_bool(&self) -> Scalar<bool> {
        Scalar::new(format!("((bool){})", self.val))
    }
    #[inline(always)]
    fn to_f16(&self) -> Scalar<half::f16> {
        Scalar::new(format!("__float2half_rn({})", self.val))
    }
    #[inline(always)]
    fn to_f32(&self) -> Scalar<f32> {
        Scalar::new(format!("((float){})", self.val))
    }
    #[inline(always)]
    fn to_f64(&self) -> Scalar<f64> {
        self.clone()
    }
    #[inline(always)]
    fn to_i8(&self) -> Scalar<i8> {
        Scalar::new(format!("((char){})", self.val))
    }
    #[inline(always)]
    fn to_i16(&self) -> Scalar<i16> {
        Scalar::new(format!("((short){})", self.val))
    }
    #[inline(always)]
    fn to_i32(&self) -> Scalar<i32> {
        Scalar::new(format!("((int){})", self.val))
    }
    #[inline(always)]
    fn to_i64(&self) -> Scalar<i64> {
        Scalar::new(format!("((long long){})", self.val))
    }
    #[inline(always)]
    fn to_u8(&self) -> Scalar<u8> {
        Scalar::new(format!("((unsigned char){})", self.val))
    }
    #[inline(always)]
    fn to_u16(&self) -> Scalar<u16> {
        Scalar::new(format!("((unsigned short){})", self.val))
    }
    #[inline(always)]
    fn to_u32(&self) -> Scalar<u32> {
        Scalar::new(format!("((unsigned int){})", self.val))
    }
    #[inline(always)]
    fn to_u64(&self) -> Scalar<u64> {
        Scalar::new(format!("((unsigned long long){})", self.val))
    }
    #[inline(always)]
    fn to_bf16(&self) -> Scalar<half::bf16> {
        Scalar::new(format!("__double2bfloat16_rn({})", self.val))
    }
    #[inline(always)]
    fn to_isize(&self) -> Scalar<isize> {
        #[cfg(target_pointer_width = "64")]
        return Scalar::new(format!("((long long){})", self.val));
        #[cfg(target_pointer_width = "32")]
        return Scalar::new(format!("((int){})", self.val));
    }
    #[inline(always)]
    fn to_usize(&self) -> Scalar<usize> {
        Scalar::new(format!("((size_t){})", self.val))
    }
    #[inline(always)]
    fn to_complex32(&self) -> Scalar<Complex32> {
        Scalar::new(format!("make_cuFloatComplex((float){}, 0.0f)", self.val))
    }
    #[inline(always)]
    fn to_complex64(&self) -> Scalar<Complex64> {
        Scalar::new(format!("make_cuDoubleComplex({}, 0.0)", self.val))
    }
}
impl CudaConvertor for Scalar<i8> {
    #[inline(always)]
    fn to_bool(&self) -> Scalar<bool> {
        Scalar::new(format!("((bool){})", self.val))
    }
    #[inline(always)]
    fn to_f16(&self) -> Scalar<half::f16> {
        Scalar::new(format!("__short2half_rn((short){})", self.val))
    }
    #[inline(always)]
    fn to_f32(&self) -> Scalar<f32> {
        Scalar::new(format!("((float){})", self.val))
    }
    #[inline(always)]
    fn to_f64(&self) -> Scalar<f64> {
        Scalar::new(format!("((double){})", self.val))
    }
    #[inline(always)]
    fn to_i8(&self) -> Scalar<i8> {
        self.clone()
    }
    #[inline(always)]
    fn to_i16(&self) -> Scalar<i16> {
        Scalar::new(format!("((short){})", self.val))
    }
    #[inline(always)]
    fn to_i32(&self) -> Scalar<i32> {
        Scalar::new(format!("((int){})", self.val))
    }
    #[inline(always)]
    fn to_i64(&self) -> Scalar<i64> {
        Scalar::new(format!("((long long){})", self.val))
    }
    #[inline(always)]
    fn to_u8(&self) -> Scalar<u8> {
        Scalar::new(format!("((unsigned char){})", self.val))
    }
    #[inline(always)]
    fn to_u16(&self) -> Scalar<u16> {
        Scalar::new(format!("((unsigned short){})", self.val))
    }
    #[inline(always)]
    fn to_u32(&self) -> Scalar<u32> {
        Scalar::new(format!("((unsigned int){})", self.val))
    }
    #[inline(always)]
    fn to_u64(&self) -> Scalar<u64> {
        Scalar::new(format!("((unsigned long long){})", self.val))
    }
    #[inline(always)]
    fn to_bf16(&self) -> Scalar<half::bf16> {
        Scalar::new(format!("__short2bfloat16_rn((short){})", self.val))
    }
    #[inline(always)]
    fn to_isize(&self) -> Scalar<isize> {
        #[cfg(target_pointer_width = "64")]
        return Scalar::new(format!("((long long){})", self.val));
        #[cfg(target_pointer_width = "32")]
        return Scalar::new(format!("((int){})", self.val));
    }
    #[inline(always)]
    fn to_usize(&self) -> Scalar<usize> {
        Scalar::new(format!("((size_t){})", self.val))
    }
    #[inline(always)]
    fn to_complex32(&self) -> Scalar<Complex32> {
        Scalar::new(format!("make_cuFloatComplex((float){}, 0.0f)", self.val))
    }
    #[inline(always)]
    fn to_complex64(&self) -> Scalar<Complex64> {
        Scalar::new(format!("make_cuDoubleComplex((double){}, 0.0)", self.val))
    }
}
impl CudaConvertor for Scalar<i16> {
    #[inline(always)]
    fn to_bool(&self) -> Scalar<bool> {
        Scalar::new(format!("((bool){})", self.val))
    }
    #[inline(always)]
    fn to_f16(&self) -> Scalar<half::f16> {
        Scalar::new(format!("__short2half_rn({})", self.val))
    }
    #[inline(always)]
    fn to_f32(&self) -> Scalar<f32> {
        Scalar::new(format!("((float){})", self.val))
    }
    #[inline(always)]
    fn to_f64(&self) -> Scalar<f64> {
        Scalar::new(format!("((double){})", self.val))
    }
    #[inline(always)]
    fn to_i8(&self) -> Scalar<i8> {
        Scalar::new(format!("((char){})", self.val))
    }
    #[inline(always)]
    fn to_i16(&self) -> Scalar<i16> {
        self.clone()
    }
    #[inline(always)]
    fn to_i32(&self) -> Scalar<i32> {
        Scalar::new(format!("((int){})", self.val))
    }
    #[inline(always)]
    fn to_i64(&self) -> Scalar<i64> {
        Scalar::new(format!("((long long){})", self.val))
    }
    #[inline(always)]
    fn to_u8(&self) -> Scalar<u8> {
        Scalar::new(format!("((unsigned char){})", self.val))
    }
    #[inline(always)]
    fn to_u16(&self) -> Scalar<u16> {
        Scalar::new(format!("((unsigned short){})", self.val))
    }
    #[inline(always)]
    fn to_u32(&self) -> Scalar<u32> {
        Scalar::new(format!("((unsigned int){})", self.val))
    }
    #[inline(always)]
    fn to_u64(&self) -> Scalar<u64> {
        Scalar::new(format!("((unsigned long long){})", self.val))
    }
    #[inline(always)]
    fn to_bf16(&self) -> Scalar<half::bf16> {
        Scalar::new(format!("__short2bfloat16_rn({})", self.val))
    }
    #[inline(always)]
    fn to_isize(&self) -> Scalar<isize> {
        #[cfg(target_pointer_width = "64")]
        return Scalar::new(format!("((long long){})", self.val));
        #[cfg(target_pointer_width = "32")]
        return Scalar::new(format!("((int){})", self.val));
    }
    #[inline(always)]
    fn to_usize(&self) -> Scalar<usize> {
        Scalar::new(format!("((size_t){})", self.val))
    }
    #[inline(always)]
    fn to_complex32(&self) -> Scalar<Complex32> {
        Scalar::new(format!("make_cuFloatComplex((float){}, 0.0f)", self.val))
    }
    #[inline(always)]
    fn to_complex64(&self) -> Scalar<Complex64> {
        Scalar::new(format!("make_cuDoubleComplex((double){}, 0.0)", self.val))
    }
}
impl CudaConvertor for Scalar<i32> {
    #[inline(always)]
    fn to_bool(&self) -> Scalar<bool> {
        Scalar::new(format!("((bool){})", self.val))
    }
    #[inline(always)]
    fn to_f16(&self) -> Scalar<half::f16> {
        Scalar::new(format!("__int2half_rn({})", self.val))
    }
    #[inline(always)]
    fn to_f32(&self) -> Scalar<f32> {
        Scalar::new(format!("((float){})", self.val))
    }
    #[inline(always)]
    fn to_f64(&self) -> Scalar<f64> {
        Scalar::new(format!("((double){})", self.val))
    }
    #[inline(always)]
    fn to_i8(&self) -> Scalar<i8> {
        Scalar::new(format!("((char){})", self.val))
    }
    #[inline(always)]
    fn to_i16(&self) -> Scalar<i16> {
        Scalar::new(format!("((short){})", self.val))
    }
    #[inline(always)]
    fn to_i32(&self) -> Scalar<i32> {
        self.clone()
    }
    #[inline(always)]
    fn to_i64(&self) -> Scalar<i64> {
        Scalar::new(format!("((long long){})", self.val))
    }
    #[inline(always)]
    fn to_u8(&self) -> Scalar<u8> {
        Scalar::new(format!("((unsigned char){})", self.val))
    }
    #[inline(always)]
    fn to_u16(&self) -> Scalar<u16> {
        Scalar::new(format!("((unsigned short){})", self.val))
    }
    #[inline(always)]
    fn to_u32(&self) -> Scalar<u32> {
        Scalar::new(format!("((unsigned int){})", self.val))
    }
    #[inline(always)]
    fn to_u64(&self) -> Scalar<u64> {
        Scalar::new(format!("((unsigned long long){})", self.val))
    }
    #[inline(always)]
    fn to_bf16(&self) -> Scalar<half::bf16> {
        Scalar::new(format!("__int2bfloat16_rn({})", self.val))
    }
    #[inline(always)]
    fn to_isize(&self) -> Scalar<isize> {
        #[cfg(target_pointer_width = "64")]
        return Scalar::new(format!("((long long){})", self.val));
        #[cfg(target_pointer_width = "32")]
        return Scalar::new(format!("((int){})", self.val));
    }
    #[inline(always)]
    fn to_usize(&self) -> Scalar<usize> {
        Scalar::new(format!("((size_t){})", self.val))
    }
    #[inline(always)]
    fn to_complex32(&self) -> Scalar<Complex32> {
        Scalar::new(format!("make_cuFloatComplex((float){}, 0.0f)", self.val))
    }
    #[inline(always)]
    fn to_complex64(&self) -> Scalar<Complex64> {
        Scalar::new(format!("make_cuDoubleComplex((double){}, 0.0)", self.val))
    }
}
impl CudaConvertor for Scalar<i64> {
    #[inline(always)]
    fn to_bool(&self) -> Scalar<bool> {
        Scalar::new(format!("((bool){})", self.val))
    }
    #[inline(always)]
    fn to_f16(&self) -> Scalar<half::f16> {
        Scalar::new(format!("__ll2half_rn({})", self.val))
    }
    #[inline(always)]
    fn to_f32(&self) -> Scalar<f32> {
        Scalar::new(format!("((float){})", self.val))
    }
    #[inline(always)]
    fn to_f64(&self) -> Scalar<f64> {
        Scalar::new(format!("((double){})", self.val))
    }
    #[inline(always)]
    fn to_i8(&self) -> Scalar<i8> {
        Scalar::new(format!("((char){})", self.val))
    }
    #[inline(always)]
    fn to_i16(&self) -> Scalar<i16> {
        Scalar::new(format!("((short){})", self.val))
    }
    #[inline(always)]
    fn to_i32(&self) -> Scalar<i32> {
        Scalar::new(format!("((int){})", self.val))
    }
    #[inline(always)]
    fn to_i64(&self) -> Scalar<i64> {
        self.clone()
    }
    #[inline(always)]
    fn to_u8(&self) -> Scalar<u8> {
        Scalar::new(format!("((unsigned char){})", self.val))
    }
    #[inline(always)]
    fn to_u16(&self) -> Scalar<u16> {
        Scalar::new(format!("((unsigned short){})", self.val))
    }
    #[inline(always)]
    fn to_u32(&self) -> Scalar<u32> {
        Scalar::new(format!("((unsigned int){})", self.val))
    }
    #[inline(always)]
    fn to_u64(&self) -> Scalar<u64> {
        Scalar::new(format!("((unsigned long long){})", self.val))
    }
    #[inline(always)]
    fn to_bf16(&self) -> Scalar<half::bf16> {
        Scalar::new(format!("__ll2bfloat16_rn({})", self.val))
    }
    #[inline(always)]
    fn to_isize(&self) -> Scalar<isize> {
        #[cfg(target_pointer_width = "64")]
        return Scalar::new(format!("((long long){})", self.val));
        #[cfg(target_pointer_width = "32")]
        return Scalar::new(format!("((int){})", self.val));
    }
    #[inline(always)]
    fn to_usize(&self) -> Scalar<usize> {
        Scalar::new(format!("((size_t){})", self.val))
    }
    #[inline(always)]
    fn to_complex32(&self) -> Scalar<Complex32> {
        Scalar::new(format!("make_cuFloatComplex((float){}, 0.0f)", self.val))
    }
    #[inline(always)]
    fn to_complex64(&self) -> Scalar<Complex64> {
        Scalar::new(format!("make_cuDoubleComplex((double){}, 0.0)", self.val))
    }
}
impl CudaConvertor for Scalar<u8> {
    #[inline(always)]
    fn to_bool(&self) -> Scalar<bool> {
        Scalar::new(format!("((bool){})", self.val))
    }
    #[inline(always)]
    fn to_f16(&self) -> Scalar<half::f16> {
        Scalar::new(format!("__ushort2half_rn((unsigned short){})", self.val))
    }
    #[inline(always)]
    fn to_f32(&self) -> Scalar<f32> {
        Scalar::new(format!("((float){})", self.val))
    }
    #[inline(always)]
    fn to_f64(&self) -> Scalar<f64> {
        Scalar::new(format!("((double){})", self.val))
    }
    #[inline(always)]
    fn to_i8(&self) -> Scalar<i8> {
        Scalar::new(format!("((char){})", self.val))
    }
    #[inline(always)]
    fn to_i16(&self) -> Scalar<i16> {
        Scalar::new(format!("((short){})", self.val))
    }
    #[inline(always)]
    fn to_i32(&self) -> Scalar<i32> {
        Scalar::new(format!("((int){})", self.val))
    }
    #[inline(always)]
    fn to_i64(&self) -> Scalar<i64> {
        Scalar::new(format!("((long long){})", self.val))
    }
    #[inline(always)]
    fn to_u8(&self) -> Scalar<u8> {
        self.clone()
    }
    #[inline(always)]
    fn to_u16(&self) -> Scalar<u16> {
        Scalar::new(format!("((unsigned short){})", self.val))
    }
    #[inline(always)]
    fn to_u32(&self) -> Scalar<u32> {
        Scalar::new(format!("((unsigned int){})", self.val))
    }
    #[inline(always)]
    fn to_u64(&self) -> Scalar<u64> {
        Scalar::new(format!("((unsigned long long){})", self.val))
    }
    #[inline(always)]
    fn to_bf16(&self) -> Scalar<half::bf16> {
        Scalar::new(format!(
            "__ushort2bfloat16_rn((unsigned short){})",
            self.val
        ))
    }
    #[inline(always)]
    fn to_isize(&self) -> Scalar<isize> {
        #[cfg(target_pointer_width = "64")]
        return Scalar::new(format!("((long long){})", self.val));
        #[cfg(target_pointer_width = "32")]
        return Scalar::new(format!("((int){})", self.val));
    }
    #[inline(always)]
    fn to_usize(&self) -> Scalar<usize> {
        Scalar::new(format!("((size_t){})", self.val))
    }
    #[inline(always)]
    fn to_complex32(&self) -> Scalar<Complex32> {
        Scalar::new(format!("make_cuFloatComplex((float){}, 0.0f)", self.val))
    }
    #[inline(always)]
    fn to_complex64(&self) -> Scalar<Complex64> {
        Scalar::new(format!("make_cuDoubleComplex((double){}, 0.0)", self.val))
    }
}
impl CudaConvertor for Scalar<u16> {
    #[inline(always)]
    fn to_bool(&self) -> Scalar<bool> {
        Scalar::new(format!("((bool){})", self.val))
    }
    #[inline(always)]
    fn to_f16(&self) -> Scalar<half::f16> {
        Scalar::<half::f16>::new(format!("__ushort2half_rn({})", self.val))
    }
    #[inline(always)]
    fn to_f32(&self) -> Scalar<f32> {
        Scalar::new(format!("((float){})", self.val))
    }
    #[inline(always)]
    fn to_f64(&self) -> Scalar<f64> {
        Scalar::new(format!("((double){})", self.val))
    }
    #[inline(always)]
    fn to_i8(&self) -> Scalar<i8> {
        Scalar::new(format!("((char){})", self.val))
    }
    #[inline(always)]
    fn to_i16(&self) -> Scalar<i16> {
        Scalar::new(format!("((short){})", self.val))
    }
    #[inline(always)]
    fn to_i32(&self) -> Scalar<i32> {
        Scalar::new(format!("((int){})", self.val))
    }
    #[inline(always)]
    fn to_i64(&self) -> Scalar<i64> {
        Scalar::new(format!("((long long){})", self.val))
    }
    #[inline(always)]
    fn to_u8(&self) -> Scalar<u8> {
        Scalar::new(format!("((unsigned char){})", self.val))
    }
    #[inline(always)]
    fn to_u16(&self) -> Scalar<u16> {
        self.clone()
    }
    #[inline(always)]
    fn to_u32(&self) -> Scalar<u32> {
        Scalar::new(format!("((unsigned int){})", self.val))
    }
    #[inline(always)]
    fn to_u64(&self) -> Scalar<u64> {
        Scalar::new(format!("((unsigned long long){})", self.val))
    }
    #[inline(always)]
    fn to_bf16(&self) -> Scalar<half::bf16> {
        Scalar::new(format!("__ushort2bfloat16_rn({})", self.val))
    }
    #[inline(always)]
    fn to_isize(&self) -> Scalar<isize> {
        #[cfg(target_pointer_width = "64")]
        return Scalar::new(format!("((long long){})", self.val));
        #[cfg(target_pointer_width = "32")]
        return Scalar::new(format!("((int){})", self.val));
    }
    #[inline(always)]
    fn to_usize(&self) -> Scalar<usize> {
        Scalar::new(format!("((size_t){})", self.val))
    }
    #[inline(always)]
    fn to_complex32(&self) -> Scalar<Complex32> {
        Scalar::new(format!("make_cuFloatComplex((float){}, 0.0f)", self.val))
    }
    #[inline(always)]
    fn to_complex64(&self) -> Scalar<Complex64> {
        Scalar::new(format!("make_cuDoubleComplex((double){}, 0.0)", self.val))
    }
}
impl CudaConvertor for Scalar<u32> {
    #[inline(always)]
    fn to_bool(&self) -> Scalar<bool> {
        Scalar::new(format!("((bool){})", self.val))
    }
    #[inline(always)]
    fn to_f16(&self) -> Scalar<half::f16> {
        Scalar::new(format!("__uint2half_rn({})", self.val))
    }
    #[inline(always)]
    fn to_f32(&self) -> Scalar<f32> {
        Scalar::new(format!("((float){})", self.val))
    }
    #[inline(always)]
    fn to_f64(&self) -> Scalar<f64> {
        Scalar::new(format!("((double){})", self.val))
    }
    #[inline(always)]
    fn to_i8(&self) -> Scalar<i8> {
        Scalar::new(format!("((char){})", self.val))
    }
    #[inline(always)]
    fn to_i16(&self) -> Scalar<i16> {
        Scalar::new(format!("((short){})", self.val))
    }
    #[inline(always)]
    fn to_i32(&self) -> Scalar<i32> {
        Scalar::new(format!("((int){})", self.val))
    }
    #[inline(always)]
    fn to_i64(&self) -> Scalar<i64> {
        Scalar::new(format!("((long long){})", self.val))
    }
    #[inline(always)]
    fn to_u8(&self) -> Scalar<u8> {
        Scalar::new(format!("((unsigned char){})", self.val))
    }
    #[inline(always)]
    fn to_u16(&self) -> Scalar<u16> {
        Scalar::new(format!("((unsigned short){})", self.val))
    }
    #[inline(always)]
    fn to_u32(&self) -> Scalar<u32> {
        self.clone()
    }
    #[inline(always)]
    fn to_u64(&self) -> Scalar<u64> {
        Scalar::new(format!("((unsigned long long){})", self.val))
    }
    #[inline(always)]
    fn to_bf16(&self) -> Scalar<half::bf16> {
        Scalar::new(format!("__uint2bfloat16_rn({})", self.val))
    }
    #[inline(always)]
    fn to_isize(&self) -> Scalar<isize> {
        #[cfg(target_pointer_width = "64")]
        return Scalar::new(format!("((long long){})", self.val));
        #[cfg(target_pointer_width = "32")]
        return Scalar::new(format!("((int){})", self.val));
    }
    #[inline(always)]
    fn to_usize(&self) -> Scalar<usize> {
        Scalar::new(format!("((size_t){})", self.val))
    }
    #[inline(always)]
    fn to_complex32(&self) -> Scalar<Complex32> {
        Scalar::new(format!("make_cuFloatComplex((float){}, 0.0f)", self.val))
    }
    #[inline(always)]
    fn to_complex64(&self) -> Scalar<Complex64> {
        Scalar::new(format!("make_cuDoubleComplex((double){}, 0.0)", self.val))
    }
}
impl CudaConvertor for Scalar<u64> {
    #[inline(always)]
    fn to_bool(&self) -> Scalar<bool> {
        Scalar::new(format!("((bool){})", self.val))
    }
    #[inline(always)]
    fn to_f16(&self) -> Scalar<half::f16> {
        Scalar::<half::f16>::new(format!("__ull2half_rn({})", self.val))
    }
    #[inline(always)]
    fn to_f32(&self) -> Scalar<f32> {
        Scalar::new(format!("((float){})", self.val))
    }
    #[inline(always)]
    fn to_f64(&self) -> Scalar<f64> {
        Scalar::new(format!("((double){})", self.val))
    }
    #[inline(always)]
    fn to_i8(&self) -> Scalar<i8> {
        Scalar::new(format!("((char){})", self.val))
    }
    #[inline(always)]
    fn to_i16(&self) -> Scalar<i16> {
        Scalar::new(format!("((short){})", self.val))
    }
    #[inline(always)]
    fn to_i32(&self) -> Scalar<i32> {
        Scalar::new(format!("((int){})", self.val))
    }
    #[inline(always)]
    fn to_i64(&self) -> Scalar<i64> {
        Scalar::new(format!("((long long){})", self.val))
    }
    #[inline(always)]
    fn to_u8(&self) -> Scalar<u8> {
        Scalar::new(format!("((unsigned char){})", self.val))
    }
    #[inline(always)]
    fn to_u16(&self) -> Scalar<u16> {
        Scalar::new(format!("((unsigned short){})", self.val))
    }
    #[inline(always)]
    fn to_u32(&self) -> Scalar<u32> {
        Scalar::new(format!("((unsigned int){})", self.val))
    }
    #[inline(always)]
    fn to_u64(&self) -> Scalar<u64> {
        self.clone()
    }
    #[inline(always)]
    fn to_bf16(&self) -> Scalar<half::bf16> {
        Scalar::<half::bf16>::new(format!("__ull2bfloat16_rn({})", self.val))
    }
    #[inline(always)]
    fn to_isize(&self) -> Scalar<isize> {
        #[cfg(target_pointer_width = "64")]
        return Scalar::new(format!("((long long){})", self.val));
        #[cfg(target_pointer_width = "32")]
        return Scalar::new(format!("((int){})", self.val));
    }
    #[inline(always)]
    fn to_usize(&self) -> Scalar<usize> {
        Scalar::new(format!("((size_t){})", self.val))
    }
    #[inline(always)]
    fn to_complex32(&self) -> Scalar<Complex32> {
        Scalar::new(format!("make_cuFloatComplex((float){}, 0.0f)", self.val))
    }
    #[inline(always)]
    fn to_complex64(&self) -> Scalar<Complex64> {
        Scalar::new(format!("make_cuDoubleComplex((double){}, 0.0)", self.val))
    }
}
impl CudaConvertor for Scalar<bf16> {
    #[inline(always)]
    fn to_bool(&self) -> Scalar<bool> {
        Scalar::new(format!("((bool)__bfloat162ushort_rn({}))", self.val))
    }
    #[inline(always)]
    fn to_f16(&self) -> Scalar<half::f16> {
        Scalar::new(format!("__float2half(__bfloat162float({}))", self.val))
    }
    #[inline(always)]
    fn to_f32(&self) -> Scalar<f32> {
        Scalar::new(format!("__bfloat162float({})", self.val))
    }
    #[inline(always)]
    fn to_f64(&self) -> Scalar<f64> {
        Scalar::new(format!("((double)__bfloat162float({}))", self.val))
    }
    #[inline(always)]
    fn to_i8(&self) -> Scalar<i8> {
        Scalar::new(format!("((char)__bfloat162short_rn({}))", self.val))
    }
    #[inline(always)]
    fn to_i16(&self) -> Scalar<i16> {
        Scalar::new(format!("((short)__bfloat162short_rn({}))", self.val))
    }
    #[inline(always)]
    fn to_i32(&self) -> Scalar<i32> {
        Scalar::new(format!("((int)__bfloat162int_rn({}))", self.val))
    }
    #[inline(always)]
    fn to_i64(&self) -> Scalar<i64> {
        Scalar::new(format!("((long long)__bfloat162ll_rn({}))", self.val))
    }
    #[inline(always)]
    fn to_u8(&self) -> Scalar<u8> {
        Scalar::new(format!(
            "((unsigned char)__bfloat162ushort_rn({}))",
            self.val
        ))
    }
    #[inline(always)]
    fn to_u16(&self) -> Scalar<u16> {
        Scalar::new(format!(
            "((unsigned short)__bfloat162ushort_rn({}))",
            self.val
        ))
    }
    #[inline(always)]
    fn to_u32(&self) -> Scalar<u32> {
        Scalar::new(format!("((unsigned int)__bfloat162uint_rn({}))", self.val))
    }
    #[inline(always)]
    fn to_u64(&self) -> Scalar<u64> {
        Scalar::new(format!(
            "((unsigned long long)__bfloat162ull_rn({}))",
            self.val
        ))
    }
    #[inline(always)]
    fn to_bf16(&self) -> Scalar<half::bf16> {
        self.clone()
    }
    #[inline(always)]
    fn to_isize(&self) -> Scalar<isize> {
        #[cfg(target_pointer_width = "64")]
        return Scalar::new(format!("((long long)__bfloat162ll_rn({}))", self.val));
        #[cfg(target_pointer_width = "32")]
        return Scalar::new(format!("((int)__bfloat162int_rn({}))", self.val));
    }
    #[inline(always)]
    fn to_usize(&self) -> Scalar<usize> {
        #[cfg(target_pointer_width = "64")]
        return Scalar::new(format!(
            "((unsigned long long)__bfloat162ull_rn({}))",
            self.val
        ));
        #[cfg(target_pointer_width = "32")]
        return Scalar::new(format!("((unsigned int)__bfloat162uint_rn({}))", self.val));
    }
    #[inline(always)]
    fn to_complex32(&self) -> Scalar<Complex32> {
        Scalar::new(format!(
            "make_cuFloatComplex((float)__bfloat162float({}), 0.0f)",
            self.val
        ))
    }
    #[inline(always)]
    fn to_complex64(&self) -> Scalar<Complex64> {
        Scalar::new(format!(
            "make_cuDoubleComplex((double)__bfloat162float({}), 0.0)",
            self.val
        ))
    }
}
impl CudaConvertor for Scalar<isize> {
    #[inline(always)]
    fn to_bool(&self) -> Scalar<bool> {
        Scalar::new(format!("((bool){})", self.val))
    }
    #[inline(always)]
    fn to_f16(&self) -> Scalar<half::f16> {
        #[cfg(target_pointer_width = "64")]
        return Scalar::<half::f16>::new(format!("__ll2half_rn({})", self.val));
        #[cfg(target_pointer_width = "32")]
        return Scalar::<half::f16>::new(format!("__int2half_rn({})", self.val));
    }
    #[inline(always)]
    fn to_f32(&self) -> Scalar<f32> {
        Scalar::new(format!("((float){})", self.val))
    }
    #[inline(always)]
    fn to_f64(&self) -> Scalar<f64> {
        Scalar::new(format!("((double){})", self.val))
    }
    #[inline(always)]
    fn to_i8(&self) -> Scalar<i8> {
        Scalar::new(format!("((char){})", self.val))
    }
    #[inline(always)]
    fn to_i16(&self) -> Scalar<i16> {
        Scalar::new(format!("((short){})", self.val))
    }
    #[inline(always)]
    fn to_i32(&self) -> Scalar<i32> {
        Scalar::new(format!("((int){})", self.val))
    }
    #[inline(always)]
    fn to_i64(&self) -> Scalar<i64> {
        Scalar::new(format!("((long long){})", self.val))
    }
    #[inline(always)]
    fn to_u8(&self) -> Scalar<u8> {
        Scalar::new(format!("((unsigned char){})", self.val))
    }
    #[inline(always)]
    fn to_u16(&self) -> Scalar<u16> {
        Scalar::new(format!("((unsigned short){})", self.val))
    }
    #[inline(always)]
    fn to_u32(&self) -> Scalar<u32> {
        Scalar::new(format!("((unsigned int){})", self.val))
    }
    #[inline(always)]
    fn to_u64(&self) -> Scalar<u64> {
        Scalar::new(format!("((unsigned long long){})", self.val))
    }
    #[inline(always)]
    fn to_bf16(&self) -> Scalar<half::bf16> {
        #[cfg(target_pointer_width = "64")]
        return Scalar::<half::bf16>::new(format!("__ll2bfloat16_rn({})", self.val));
        #[cfg(target_pointer_width = "32")]
        return Scalar::<half::bf16>::new(format!("__int2bfloat16_rn({})", self.val));
    }
    #[inline(always)]
    fn to_isize(&self) -> Scalar<isize> {
        self.clone()
    }
    #[inline(always)]
    fn to_usize(&self) -> Scalar<usize> {
        Scalar::new(format!("((size_t){})", self.val))
    }
    #[inline(always)]
    fn to_complex32(&self) -> Scalar<Complex32> {
        Scalar::new(format!("make_cuFloatComplex((float){}, 0.0f)", self.val))
    }
    #[inline(always)]
    fn to_complex64(&self) -> Scalar<Complex64> {
        Scalar::new(format!("make_cuDoubleComplex((double){}, 0.0)", self.val))
    }
}
impl CudaConvertor for Scalar<usize> {
    #[inline(always)]
    fn to_bool(&self) -> Scalar<bool> {
        Scalar::new(format!("((bool){})", self.val))
    }
    #[inline(always)]
    fn to_f16(&self) -> Scalar<half::f16> {
        #[cfg(target_pointer_width = "64")]
        return Scalar::<half::f16>::new(format!("__ull2half_rn({})", self.val));
        #[cfg(target_pointer_width = "32")]
        return Scalar::<half::f16>::new(format!("__uint2half_rn({})", self.val));
    }
    #[inline(always)]
    fn to_f32(&self) -> Scalar<f32> {
        Scalar::new(format!("((float){})", self.val))
    }
    #[inline(always)]
    fn to_f64(&self) -> Scalar<f64> {
        Scalar::new(format!("((double){})", self.val))
    }
    #[inline(always)]
    fn to_i8(&self) -> Scalar<i8> {
        Scalar::new(format!("((char){})", self.val))
    }
    #[inline(always)]
    fn to_i16(&self) -> Scalar<i16> {
        Scalar::new(format!("((short){})", self.val))
    }
    #[inline(always)]
    fn to_i32(&self) -> Scalar<i32> {
        Scalar::new(format!("((int){})", self.val))
    }
    #[inline(always)]
    fn to_i64(&self) -> Scalar<i64> {
        Scalar::new(format!("((long long){})", self.val))
    }
    #[inline(always)]
    fn to_u8(&self) -> Scalar<u8> {
        Scalar::new(format!("((unsigned char){})", self.val))
    }
    #[inline(always)]
    fn to_u16(&self) -> Scalar<u16> {
        Scalar::new(format!("((unsigned short){})", self.val))
    }
    #[inline(always)]
    fn to_u32(&self) -> Scalar<u32> {
        Scalar::new(format!("((unsigned int){})", self.val))
    }
    #[inline(always)]
    fn to_u64(&self) -> Scalar<u64> {
        Scalar::new(format!("((unsigned long long){})", self.val))
    }
    #[inline(always)]
    fn to_bf16(&self) -> Scalar<half::bf16> {
        #[cfg(target_pointer_width = "64")]
        return Scalar::new(format!("__ull2bfloat16_rn({})", self.val));
        #[cfg(target_pointer_width = "32")]
        return Scalar::new(format!("__uint2bfloat16_rn({})", self.val));
    }
    #[inline(always)]
    fn to_isize(&self) -> Scalar<isize> {
        Scalar::new(format!("((long long){})", self.val))
    }
    #[inline(always)]
    fn to_usize(&self) -> Scalar<usize> {
        self.clone()
    }
    #[inline(always)]
    fn to_complex32(&self) -> Scalar<Complex32> {
        Scalar::new(format!("make_cuFloatComplex((float){}, 0.0f)", self.val))
    }
    #[inline(always)]
    fn to_complex64(&self) -> Scalar<Complex64> {
        Scalar::new(format!("make_cuDoubleComplex((double){}, 0.0)", self.val))
    }
}
impl CudaConvertor for Scalar<Complex32> {
    #[inline(always)]
    fn to_bool(&self) -> Scalar<bool> {
        Scalar::new(format!(
            "(bool)(cuCrealf({}) != 0.0f || cuCimagf({}) != 0.0f)",
            self.val, self.val
        ))
    }
    #[inline(always)]
    fn to_f16(&self) -> Scalar<f16> {
        Scalar::new(format!("__float2half(cuCrealf({}))", self.val))
    }
    #[inline(always)]
    fn to_f32(&self) -> Scalar<f32> {
        Scalar::new(format!("cuCrealf({})", self.val))
    }
    #[inline(always)]
    fn to_f64(&self) -> Scalar<f64> {
        Scalar::new(format!("(double)cuCrealf({})", self.val))
    }
    #[inline(always)]
    fn to_i8(&self) -> Scalar<i8> {
        Scalar::new(format!("((char)cuCrealf({}))", self.val))
    }
    #[inline(always)]
    fn to_i16(&self) -> Scalar<i16> {
        Scalar::new(format!("((short)cuCrealf({}))", self.val))
    }
    #[inline(always)]
    fn to_i32(&self) -> Scalar<i32> {
        Scalar::new(format!("((int)cuCrealf({}))", self.val))
    }
    #[inline(always)]
    fn to_i64(&self) -> Scalar<i64> {
        Scalar::new(format!("((long long)cuCrealf({}))", self.val))
    }
    #[inline(always)]
    fn to_u8(&self) -> Scalar<u8> {
        Scalar::new(format!("((unsigned char)cuCrealf({}))", self.val))
    }
    #[inline(always)]
    fn to_u16(&self) -> Scalar<u16> {
        Scalar::new(format!("((unsigned short)cuCrealf({}))", self.val))
    }
    #[inline(always)]
    fn to_u32(&self) -> Scalar<u32> {
        Scalar::new(format!("((unsigned int)cuCrealf({}))", self.val))
    }
    #[inline(always)]
    fn to_u64(&self) -> Scalar<u64> {
        Scalar::new(format!("((unsigned long long)cuCrealf({}))", self.val))
    }
    #[inline(always)]
    fn to_bf16(&self) -> Scalar<half::bf16> {
        Scalar::new(format!("__float2bfloat16_rn(cuCrealf({}))", self.val))
    }
    #[inline(always)]
    fn to_isize(&self) -> Scalar<isize> {
        Scalar::new(format!("((size_t)cuCrealf({}))", self.val))
    }
    #[inline(always)]
    fn to_usize(&self) -> Scalar<usize> {
        Scalar::new(format!("((size_t)cuCrealf({}))", self.val))
    }
    #[inline(always)]
    fn to_complex32(&self) -> Scalar<Complex32> {
        self.clone()
    }
    #[inline(always)]
    fn to_complex64(&self) -> Scalar<Complex64> {
        Scalar::new(format!(
            "make_cuDoubleComplex((double)cuCrealf({}), (double)cuCimagf({}))",
            self.val, self.val
        ))
    }
}
impl CudaConvertor for Scalar<Complex64> {
    #[inline(always)]
    fn to_bool(&self) -> Scalar<bool> {
        Scalar::new(format!(
            "(bool)(cuCreal({}) != 0.0 || cuCimag({}) != 0.0)",
            self.val, self.val
        ))
    }
    #[inline(always)]
    fn to_f16(&self) -> Scalar<half::f16> {
        Scalar::new(format!("__float2half((float)cuCreal({}))", self.val))
    }
    #[inline(always)]
    fn to_f32(&self) -> Scalar<f32> {
        Scalar::new(format!("(float)cuCreal({})", self.val))
    }
    #[inline(always)]
    fn to_f64(&self) -> Scalar<f64> {
        Scalar::new(format!("cuCreal({})", self.val))
    }
    #[inline(always)]
    fn to_i8(&self) -> Scalar<i8> {
        Scalar::new(format!("((char)cuCreal({}))", self.val))
    }
    #[inline(always)]
    fn to_i16(&self) -> Scalar<i16> {
        Scalar::new(format!("((short)cuCreal({}))", self.val))
    }
    #[inline(always)]
    fn to_i32(&self) -> Scalar<i32> {
        Scalar::new(format!("((int)cuCreal({}))", self.val))
    }
    #[inline(always)]
    fn to_i64(&self) -> Scalar<i64> {
        Scalar::new(format!("((long long)cuCreal({}))", self.val))
    }
    #[inline(always)]
    fn to_u8(&self) -> Scalar<u8> {
        Scalar::new(format!("((unsigned char)cuCreal({}))", self.val))
    }
    #[inline(always)]
    fn to_u16(&self) -> Scalar<u16> {
        Scalar::new(format!("((unsigned short)cuCreal({}))", self.val))
    }
    #[inline(always)]
    fn to_u32(&self) -> Scalar<u32> {
        Scalar::new(format!("((unsigned int)cuCreal({}))", self.val))
    }
    #[inline(always)]
    fn to_u64(&self) -> Scalar<u64> {
        Scalar::new(format!("((unsigned long long)cuCreal({}))", self.val))
    }
    #[inline(always)]
    fn to_bf16(&self) -> Scalar<half::bf16> {
        Scalar::new(format!("__float2bfloat16_rn((float)cuCreal({}))", self.val))
    }
    #[inline(always)]
    fn to_isize(&self) -> Scalar<isize> {
        Scalar::new(format!("((size_t)cuCreal({}))", self.val))
    }
    #[inline(always)]
    fn to_usize(&self) -> Scalar<usize> {
        Scalar::new(format!("((size_t)cuCreal({}))", self.val))
    }
    #[inline(always)]
    fn to_complex32(&self) -> Scalar<Complex32> {
        Scalar::new(format!(
            "make_cuFloatComplex((float)cuCreal({}), (float)cuCimag({}))",
            self.val, self.val
        ))
    }
    #[inline(always)]
    fn to_complex64(&self) -> Scalar<Complex64> {
        self.clone()
    }
}
