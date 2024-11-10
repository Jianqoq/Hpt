#[derive(Clone, Eq, PartialEq, Hash, Copy, Debug)]
pub(crate) enum KernelType {
    Unary,
    Binary,
}

impl KernelType {
    pub fn infer_suc_kernel(&self, next: &KernelType) -> Option<KernelType> {
        match (self, next) {
            (KernelType::Unary, KernelType::Unary) => Some(KernelType::Unary),
            (KernelType::Binary, KernelType::Binary) => Some(KernelType::Binary),
            (KernelType::Binary, KernelType::Unary) => Some(KernelType::Binary),
            (KernelType::Unary, KernelType::Binary) => Some(KernelType::Binary),
        }
    }
    pub fn infer_pred_kernel(&self, pred: &KernelType) -> Option<KernelType> {
        match (self, pred) {
            (KernelType::Unary, KernelType::Unary) => Some(KernelType::Unary),
            (KernelType::Binary, KernelType::Binary) => Some(KernelType::Binary),
            (KernelType::Binary, KernelType::Unary) => Some(KernelType::Binary),
            (KernelType::Unary, KernelType::Binary) => Some(KernelType::Binary),
        }
    }
}
