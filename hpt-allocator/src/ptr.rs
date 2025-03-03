/// just a wrapper around `*mut u8`, implementing `Send` and `Sync` trait to let the compiler know that it is safe to send and share across threads
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub(crate) struct SafePtr {
    pub(crate) ptr: *mut u8,
}
unsafe impl Send for SafePtr {}
unsafe impl Sync for SafePtr {}
