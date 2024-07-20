use tensor_compiler::{
    halide::{ code_gen::code_gen::CodeGen, module::Module, printer::IRPrinter, variable::Variable },
    hlir::schedule::schedule::Schedule,
};
use tensor_dyn::tensor_base::_Tensor;
use tensor_llvm::context::context::Context;
use tensor_traits::random::Random;
use tensor_types::dtype::Dtype;

// pub fn main() -> anyhow::Result<()> {
//     let a = _Tensor::<f32>::randn(&[1, 3])?;
//     let b = _Tensor::<f32>::randn(&[3, 1])?;

//     let d = a
//         .iter()
//         .zip(b.iter())
//         .strided_map(|(x, y)| x * y)
//         .collect::<_Tensor<f32>>();

//     println!("{}", d);
//     Ok(())
// }

fn main() -> anyhow::Result<()> {
    // let m = Variable::make("m");
    // let n = Variable::make("n");
    // let o = Variable::make("o");

    // let a = tensor_compiler::hlir::tensor::Tensor::placeholder(&[&n, &m, &o], Dtype::F32, "A");
    // let b = a.sum(0f32, 1);

    // let d = b.reshape(&[&n, &1i64, &o]);

    // let c = &a - &d;

    // let e = d.sum(0f32, 1);

    // let s = Schedule::create(&[&a, &d, &c, &e, &b]);

    // let lowered = s.lower("main");
    // let mut module = Module::new("main");
    // module.add_function(lowered.ty, &lowered.name);
    // module.get_function_mut(&lowered.name).unwrap().body = lowered.body;
    // IRPrinter.print_module(&module);

    // let tensor_a = tensor_dyn::tensor::Tensor::<f32>
    //     ::arange(0f32, 100f32)
    //     .unwrap()
    //     .reshape(&[2, 2, 5, 5])
    //     .unwrap();
    // let tensor_c = tensor_dyn::tensor::Tensor::<f32>::empty(&[2, 5, 5]).unwrap();
    // let tensor_d = tensor_dyn::tensor::Tensor::<f32>::empty(&[2, 5]).unwrap();
    // let exec_a = tensor_compiler::tensor::Tensor::new(tensor_a.clone().into(), a.name());
    // let exec_c = tensor_compiler::tensor::Tensor::new(tensor_c.clone().into(), c.name());
    // let exec_d = tensor_compiler::tensor::Tensor::new(tensor_d.clone().into(), d.name());
    // executable.run(&[exec_a], &[exec_d], &[exec_c]);
    // println!("{}", tensor_c);
    Ok(())
}
