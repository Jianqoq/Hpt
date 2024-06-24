
#[derive(Clone, Copy)]
pub struct Cpu;

#[derive(Clone, Copy)]
pub struct Cuda;

#[derive(Clone, Copy)]
pub struct Backend<B> {
    _backend: B,
}

pub trait TensorBackend {
    fn new() -> Self;
}

impl TensorBackend for Backend<Cpu> {
    fn new() -> Self {
        Backend { _backend: Cpu }
    }
}

impl TensorBackend for Backend<Cuda> {
    fn new() -> Self {
        Backend { _backend: Cuda }
    }
}