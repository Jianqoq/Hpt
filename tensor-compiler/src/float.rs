use std::mem;

use serde::Serialize;

#[derive(Clone, Debug, PartialEq, PartialOrd, Copy, Eq, Hash)]
pub struct Float(u64, i16, i8);

impl Serialize for Float {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_f64(self.to_f64())
    }
}

impl<'de> serde::Deserialize<'de> for Float {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let val = f64::deserialize(deserializer)?;
        Ok(val.into())
    }
}

impl Float {
    pub fn new(num: f64) -> Self {
        let res = Self::integer_decode(num);
        Self(res.0, res.1, res.2)
    }

    // source code from rust stdlib
    pub fn integer_decode(val: f64) -> (u64, i16, i8) {
        let bits: u64 = unsafe { mem::transmute(val) };
        let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
        let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
        let mantissa = if exponent == 0 {
            (bits & 0xfffffffffffff) << 1
        } else {
            (bits & 0xfffffffffffff) | 0x10000000000000
        };

        exponent -= 1023 + 52;
        (mantissa, exponent, sign)
    }

    // source code from rust stdlib
    pub fn to_f64(&self) -> f64 {
        let (mantissa, exponent, sign) = (self.0, self.1, self.2);
        let sign_bit = if sign == -1 { 1u64 << 63 } else { 0 };
        let exponent_bits = ((exponent + 1023 + 52) as u64) << 52;
        let mantissa_bits = mantissa & 0x000f_ffff_ffff_ffff;

        let bits = sign_bit | exponent_bits | mantissa_bits;
        unsafe { mem::transmute(bits) }
    }
}

impl From<f64> for Float {
    fn from(val: f64) -> Self {
        Self::new(val)
    }
}