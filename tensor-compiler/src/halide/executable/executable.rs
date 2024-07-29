use std::{ alloc::Layout, collections::HashMap, ffi::c_void, sync::Arc };

use tensor_llvm::{
    builder::builder::Builder,
    context::context::Context,
    engine::engine::ExecutionEngine,
    utils::to_c_str,
};
use tensor_types::dtype::Dtype;

use crate::{ halide::prime_expr::PrimeExpr, te::hstrides::HStrides };

use super::arr::Array;

pub struct Executable {
    pub ctx: Context,
    pub module: tensor_llvm::module::module::Module,
    pub builder: Builder,
    pub ee: ExecutionEngine,
    pub sorted_fns: Vec<Arc<String>>,
    pub sorted_strides_cal: Vec<Arc<dyn Fn(&HashMap<Arc<String>, i64>) -> Vec<HStrides>>>,
    pub sorted_inputs: Vec<Vec<usize>>,
    pub sorted_outputs: Vec<Vec<usize>>,
    pub inps_shapes: Vec<Vec<Arc<Vec<i64>>>>,
    pub inps_dtypes: Vec<Vec<Dtype>>,
    pub outs_shapes: Vec<Vec<Arc<Vec<i64>>>>,
    pub outs_dtypes: Vec<Vec<Dtype>>,
    pub sorted_inputs_container: Vec<*mut *mut c_void>,
    pub sorted_outputs_container: Vec<*mut *mut c_void>,
    pub sorted_ic_layouts: Vec<Vec<Layout>>,
    pub sorted_oc_layouts: Vec<Vec<Layout>>,
    pub sorted_touse_vars: Vec<Vec<Arc<String>>>,
    pub sorted_out_shapes: Vec<Vec<Arc<Vec<PrimeExpr>>>>,
    pub touse_vars_layout: Vec<Layout>,
    pub istrides_vec_layout: Vec<Layout>,
    pub ostrides_vec_layout: Vec<Layout>,
    pub strides_layout: Vec<Vec<Layout>>,
    pub ostrides_layouts: Vec<Vec<Layout>>,
    pub vars_map: HashMap<Arc<String>, i64>,
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

    pub fn execute(&self, inputs: HashMap<usize, Array>, outputs: HashMap<usize, Array>) {
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
            self.sorted_inputs[idx]
                .iter()
                .enumerate()
                .for_each(|(i, x)| {
                    let inp_shape = inputs[x].shape().inner();
                    let expect_shape = self.inps_shapes[idx][i].as_ref();
                    if inp_shape != expect_shape {
                        panic!(
                            "input shape mismatch: expect {:?}, got {:?}, input index: {}",
                            expect_shape,
                            inp_shape,
                            x
                        );
                    }
                    let inp_dtype = inputs[x].dtype();
                    let expect_dtype = self.inps_dtypes[idx][i];
                    if inp_dtype != expect_dtype {
                        panic!(
                            "input dtype mismatch: expect {:?}, got {:?}, input index: {}",
                            expect_dtype,
                            inp_dtype,
                            x
                        );
                    }
                    let data = inputs[x].ptr();
                    unsafe {
                        self.sorted_inputs_container[idx].add(i).write(data);
                    }
                });
            self.sorted_outputs[idx]
                .iter()
                .enumerate()
                .for_each(|(i, x)| {
                    let out_shape = outputs[x].shape().inner();
                    let expect_shape = self.outs_shapes[idx][i].as_ref();
                    if out_shape != expect_shape {
                        panic!(
                            "output shape mismatch: expect {:?}, got {:?}, output index: {}",
                            expect_shape,
                            out_shape,
                            x
                        );
                    }
                    let out_dtype = outputs[x].dtype();
                    let expect_dtype = self.outs_dtypes[idx][i];
                    if out_dtype != expect_dtype {
                        panic!(
                            "output dtype mismatch: expect {:?}, got {:?}, output index: {}",
                            expect_dtype,
                            out_dtype,
                            x
                        );
                    }
                    let data = outputs[x].ptr();
                    unsafe {
                        self.sorted_outputs_container[idx].add(i).write(data);
                    }
                });
            let input_container = self.sorted_inputs_container[idx];
            let output_container = self.sorted_outputs_container[idx];
            unsafe {
                llvm_fn(
                    istrides,
                    ostrides,
                    input_container,
                    output_container,
                    std::ptr::null_mut(),
                    vars,
                    0
                );
            }
        }
    }
}
