use std::{ alloc::Layout, collections::HashMap, ffi::c_void, sync::Arc };

use tensor_common::strides_utils::shape_to_strides;
use tensor_llvm::{
    builder::builder::Builder,
    context::context::Context,
    engine::engine::ExecutionEngine,
    utils::to_c_str,
};

use crate::{
    halide::prime_expr::PrimeExpr,
    te::{ hstrides::HStrides, idx_evaluator::IdxEvaluator },
};

pub struct Executable {
    pub ctx: Context,
    pub module: tensor_llvm::module::module::Module,
    pub builder: Builder,
    pub ee: ExecutionEngine,
    pub sorted_fns: Vec<Arc<String>>,
    pub sorted_strides_cal: Vec<Arc<dyn Fn(&HashMap<Arc<String>, i64>) -> Vec<HStrides>>>,
    pub sorted_touse_vars: Vec<Vec<Arc<String>>>,
    pub sorted_out_shapes: Vec<Vec<Arc<Vec<PrimeExpr>>>>,
    pub touse_vars_layout: Vec<Layout>,
    pub istrides_vec_layout: Vec<Layout>,
    pub ostrides_vec_layout: Vec<Layout>,
    pub strides_layout: Vec<Vec<Layout>>,
    pub ostrides_layouts: Vec<Vec<Layout>>,
    pub istrides: Vec<*mut *mut i64>,
    pub ostrides: Vec<*mut *mut i64>,
    pub vars: Vec<*mut i64>,
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

    pub fn prepare(mut self, var_map: HashMap<Arc<String>, i64>) -> Self {
        let mut touse_vars_layout = vec![];
        let mut istrides_vec_layout = vec![];
        let mut ostrides_vec_layout = vec![];
        let mut strides_layout = vec![];
        let mut ostrides_layouts = vec![];
        let mut istrides = vec![];
        let mut ostrides = vec![];
        let mut vars = vec![];
        for (idx, strides_cal) in self.sorted_strides_cal.iter().enumerate() {
            let _istrides = {
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
                strides_layout.push(cached_layout);
                istrides_vec_layout.push(layout);
                strides_vec
            };

            let mut _layouts = vec![];
            let _ostrides = {
                let real_shapes = self.sorted_out_shapes[idx]
                    .iter()
                    .map(|x| {
                        x.iter()
                            .map(|x| IdxEvaluator::new(&var_map).eval(x))
                            .collect::<Vec<i64>>()
                    })
                    .collect::<Vec<_>>();
                let ptrs_layout = std::alloc::Layout
                    ::from_size_align(real_shapes.len() * std::mem::size_of::<*mut i64>(), 8)
                    .unwrap();
                let ptrs = unsafe {
                    let ptrs = std::alloc::alloc(ptrs_layout.clone()) as *mut *mut i64;
                    for (idx, s) in real_shapes.iter().enumerate() {
                        let layout = std::alloc::Layout
                            ::from_size_align(s.len() * std::mem::size_of::<i64>(), 8)
                            .unwrap();
                        let ptr = std::alloc::alloc(layout) as *mut i64;
                        for (idx, s) in s.iter().enumerate() {
                            ptr.add(idx).write(*s);
                        }
                        ptrs.add(idx).write(ptr);
                        _layouts.push(layout);
                    }
                    ptrs
                };
                ostrides_vec_layout.push(ptrs_layout);
                ptrs
            };
            ostrides_layouts.push(_layouts);

            let _vars = {
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
                touse_vars_layout.push(layout);
                shape_vars
            };
            istrides.push(_istrides);
            ostrides.push(_ostrides);
            vars.push(_vars);
        }
        self.touse_vars_layout = touse_vars_layout;
        self.istrides_vec_layout = istrides_vec_layout;
        self.ostrides_vec_layout = ostrides_vec_layout;
        self.strides_layout = strides_layout;
        self.ostrides_layouts = ostrides_layouts;
        self.istrides = istrides;
        self.ostrides = ostrides;
        self.vars = vars;
        self
    }

    pub fn execute(&self) {
        for (idx, fn_name) in self.sorted_fns.iter().enumerate() {
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
                        *mut *mut c_void /*output_vec */,
                        *mut i64 /*offset_vec */,
                        *mut i64 /*shape_vars */,
                        i64 /*thread_idx */
                    )
                >(&address)
            };
            let istrides = self.istrides[idx];
            let ostrides = self.ostrides[idx];
            let vars = self.vars[idx];
        }
    }
}
