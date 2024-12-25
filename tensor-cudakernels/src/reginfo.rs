#[derive(Debug, Clone, Copy)]
pub struct RegisterInfo {
    pub pred: usize,
    pub b16: usize,
    pub b32: usize,
    pub b64: usize,
}