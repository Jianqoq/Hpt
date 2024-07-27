use std::{ alloc::Layout, collections::HashMap, ffi::c_void, sync::Arc };

use tensor_llvm::{
    builder::builder::Builder,
    context::context::Context,
    engine::engine::ExecutionEngine,
    utils::to_c_str,
};

use crate::te::hstrides::HStrides;

pub struct Executable {
    pub ctx: Context,
    pub module: tensor_llvm::module::module::Module,
    pub builder: Builder,
    pub ee: ExecutionEngine,
    pub sorted_fns: Vec<Arc<String>>,
    pub sorted_strides_cal: Vec<Arc<dyn Fn(&HashMap<String, i64>) -> Vec<HStrides>>>,
    pub cached_strides: Vec<Option<*mut *mut i64>>,
    pub cached_vars: Vec<Option<(*mut i64, usize)>>,
    pub cached_strides_layout: Vec<Option<Vec<Layout>>>,
    pub strides_vec_layout: Vec<Option<Layout>>,
    pub sorted_touse_vars: Vec<Vec<String>>,
    pub touse_vars: Vec<Option<*mut i64>>,
    pub touse_vars_layout: Vec<Option<Layout>>,
}

impl Executable {
    pub fn print_to_file(&self, path: &str) {
        self.module.print_to_file(path).expect("failed to print to file");
    }

    pub fn get_function<F>(&self, name: &str) -> F {
        let c_str = to_c_str(name);
        let address = unsafe {
            llvm_sys::execution_engine::LLVMGetFunctionAddress(self.ee.inner(), c_str.as_ptr())
        };
        if address == 0 {
            panic!("{}", &format!("function {} not found", name));
        }
        unsafe { std::mem::transmute_copy::<u64, F>(&address) }
    }

    pub fn execute(&mut self, var_map: HashMap<String, i64>) {
        for (
            ((((idx, fn_name), strides_cal), cached_strides), cached_vars),
            touse_vars,
        ) in self.sorted_fns
            .iter()
            .enumerate()
            .zip(self.sorted_strides_cal.iter())
            .zip(self.cached_strides.iter_mut())
            .zip(self.cached_vars.iter_mut())
            .zip(self.touse_vars.iter_mut()) {
            let istrides = if let Some(strides) = cached_strides {
                *strides
            } else {
                let strides = strides_cal(&var_map);
                let mut cached_layout = vec![];
                let layout = std::alloc::Layout
                    ::from_size_align(strides.len() * std::mem::size_of::<*mut i64>(), 8)
                    .unwrap();
                let strides_vec = unsafe {
                    let strides_vec = std::alloc::alloc(layout.clone()) as *mut *mut i64;
                    for (idx, s) in strides.iter().enumerate() {
                        let s_layout = std::alloc::Layout
                            ::from_size_align(s.strides.len() * std::mem::size_of::<i64>(), 8)
                            .unwrap();
                        cached_layout.push(s_layout);
                        strides_vec.add(idx).write(s.to_aligned_ptr());
                    }
                    strides_vec
                };
                self.cached_strides_layout.get_mut(idx).unwrap().replace(cached_layout);
                self.strides_vec_layout.get_mut(idx).unwrap().replace(layout);
                *cached_strides = Some(strides_vec);
                strides_vec
            };

            let vars = if let Some(var) = touse_vars {
                *var
            } else {
                let vars = &self.sorted_touse_vars[idx];
                let layout = std::alloc::Layout
                    ::from_size_align(vars.len() * std::mem::size_of::<i64>(), 8)
                    .unwrap();
                let shape_vars = unsafe {
                    let shape_vars = std::alloc::alloc(layout.clone()) as *mut i64;
                    for (idx, var) in vars.iter().enumerate() {
                        shape_vars.add(idx).write(var_map[var]);
                    }
                    shape_vars
                };
                self.touse_vars_layout.get_mut(idx).unwrap().replace(layout);
                *touse_vars = Some(shape_vars);
                shape_vars
            };

            let c_str = to_c_str(fn_name);
            let address = unsafe {
                llvm_sys::execution_engine::LLVMGetFunctionAddress(self.ee.inner(), c_str.as_ptr())
            };
            if address == 0 {
                panic!("{}", &format!("function {} not found", fn_name));
            }
            let llvm_fn = unsafe {
                std::mem::transmute_copy::<
                    u64,
                    unsafe extern "C" fn(
                        *mut *mut i64 /* istrides_vec*/,
                        *mut *mut i64 /* ostrides_vec*/,
                        *mut *mut c_void /* data_vec*/,
                        *mut *mut c_void/*output_vec */,
                        *mut i64/*offset_vec */,
                        *mut i64/*shape_vars */,
                        i64/*thread_idx */
                    )
                >(&address)
            };
        }
    }
}
