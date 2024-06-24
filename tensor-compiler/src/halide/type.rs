#[repr(u8)]
#[derive(Clone, Hash, PartialEq, Eq, Copy, PartialOrd, Ord, Debug)]
pub enum HalideirTypeCode {
    Int = 0,
    UInt = 1,
    Float = 2,
    Handle = 3,
}

#[derive(Clone, Hash, PartialEq, Eq, Copy, Debug)]
struct HalideirType {
    code: HalideirTypeCode,
    bits: u8,
    lanes: u16,
}

impl Default for HalideirType {
    fn default() -> Self {
        Self {
            code: HalideirTypeCode::Int,
            bits: 0,
            lanes: 0,
        }
    }
}

impl HalideirType {
    const fn new(code: HalideirTypeCode, bits: u8, lanes: u16) -> Self {
        HalideirType { code, bits, lanes }
    }

    const fn bytes(&self) -> usize {
        ((self.bits as usize) + 7) / 8
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Copy, Debug)]
pub struct Type {
    halideir_type: HalideirType,
}

impl Type {
    pub const fn new(code: HalideirTypeCode, bits: u8, lanes: u16) -> Self {
        Self {
            halideir_type: HalideirType::new(code, bits, lanes),
        }
    }

    pub const fn bytes(&self) -> usize {
        self.halideir_type.bytes()
    }

    pub const fn bits(&self) -> u8 {
        self.halideir_type.bits
    }

    pub const fn lanes(&self) -> u16 {
        self.halideir_type.lanes
    }

    pub const fn code(&self) -> HalideirTypeCode {
        self.halideir_type.code
    }

    pub fn is_bool(&self) -> bool {
        self.halideir_type.code == HalideirTypeCode::UInt && self.halideir_type.bits == 1
    }

    pub fn is_int(&self) -> bool {
        self.halideir_type.code == HalideirTypeCode::Int
    }

    pub fn is_uint(&self) -> bool {
        self.halideir_type.code == HalideirTypeCode::UInt
    }

    pub fn is_float(&self) -> bool {
        self.halideir_type.code == HalideirTypeCode::Float
    }

    pub fn is_scalar(&self) -> bool {
        self.halideir_type.lanes == 1
    }

    pub fn is_vector(&self) -> bool {
        self.halideir_type.lanes > 1
    }

    pub fn is_handle(&self) -> bool {
        self.halideir_type.code == HalideirTypeCode::Handle
    }
}
