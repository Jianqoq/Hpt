pub static CUDA_STORAGE: Lazy<
    Mutex<HashMap<usize, HashMap<crate::cuda_allocator::SafePtr, usize>>>,
> = Lazy::new(|| HashMap::new().into());
