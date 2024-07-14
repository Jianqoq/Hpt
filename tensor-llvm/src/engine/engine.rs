use llvm_sys::{ execution_engine::LLVMOpaqueExecutionEngine, target::LLVMTargetDataRef };

use crate::utils::to_c_str;

pub struct ExecutionEngine {
    pub(crate) engine: *mut LLVMOpaqueExecutionEngine,
    pub(crate) target_data: LLVMTargetDataRef,
}

impl ExecutionEngine {
    pub fn inner(&self) -> *mut LLVMOpaqueExecutionEngine {
        self.engine
    }
    pub fn new(engine: *mut LLVMOpaqueExecutionEngine, target_data: LLVMTargetDataRef) -> Self {
        ExecutionEngine { engine, target_data }
    }

    pub fn get_function(
        &self,
        name: &str
    ) -> unsafe extern "C" fn(usize, *const isize, *mut *mut u8) {
        let c_str = to_c_str(name);
        let address = unsafe {
            llvm_sys::execution_engine::LLVMGetFunctionAddress(self.engine, c_str.as_ptr())
        };
        if address == 0 {
            panic!("{}", &format!("function {} not found", name));
        }
        unsafe {
            std::mem::transmute_copy::<
                u64,
                unsafe extern "C" fn(usize, *const isize, *mut *mut u8)
            >(&address)
        }
    }
}
