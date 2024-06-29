use std::fmt::Display;

use hashbrown::{ HashMap, HashSet };
use tensor_types::dtype::Dtype;

use crate::halide::{ exprs::{ Int, Load }, prime_expr::PrimeExpr, variable::Variable };

use super::exprs::Tensor;

pub struct HlirLower {
    /// this ordered exprs is the order of computation
    ordered_exprs: Vec<(PrimeExpr, PrimeExpr)>,
    /// this unique exprs is the unique exprs in the computation
    unique_exprs: HashSet<PrimeExpr>,
    /// in compilation, each node could have multiple different compute
    /// so we need to count the number of each node so that we can generate unique name for each expr
    cnts: HashMap<usize, usize>,
    /// a stack to get the for loop variables correctly
    var_stack: Vec<Variable>,
}

impl HlirLower {
    pub fn new() -> Self {
        Self {
            ordered_exprs: vec![],
            unique_exprs: HashSet::new(),
            cnts: HashMap::new(),
            var_stack: vec![],
        }
    }

    pub fn lower<T: Into<Tensor>>(&mut self, output: T) {
        let tensor: &Tensor = &output.into();
        let expr = self._lower(tensor, true);
        if !self.unique_exprs.contains(&expr) {
            self.unique_exprs.insert(expr.clone());
            let variable_name = Variable::from(format!("%{}", tensor.id()));
            self.ordered_exprs.push((variable_name.into(), expr.clone()));
        }
    }

    pub fn _lower(&mut self, output: &Tensor, push_vars: bool) -> PrimeExpr {
        match output {
            Tensor::Base(base) => {
                let mut strides = base.layout().strides().to_vec();
                let reduced_strides = base.reduced_strides().to_vec();
                strides.extend(reduced_strides.iter());
                assert!(self.var_stack.len() == strides.len());
                let variable_name = Variable::from(format!("%{}", base.id()));
                let indices = self.var_stack
                    .iter()
                    .zip(strides.iter())
                    .map(|(v, s)| v * Int::make(Dtype::I64, *s))
                    .reduce(|a, b| a + b);
                let load = Load::make(variable_name, indices.unwrap());
                load.into()
            }
            Tensor::Reduce(reduce) => {
                let mut exprs = vec![];
                for var in reduce.reduce_vars().iter() {
                    self.var_stack.push(var.clone());
                }
                for input in reduce.inputs().iter() {
                    exprs.push(self._lower(input, false));
                }
                for _ in reduce.reduce_vars().iter() {
                    self.var_stack.pop();
                }
                let call = reduce.func().call_common(exprs);
                call.into()
            }
            Tensor::Fuse(fuse) => {
                if push_vars {
                    for i in 0..fuse.ndim() {
                        self.var_stack.push(Variable::new(format!("i{}", i)));
                    }
                }
                let mut exprs = vec![];
                for input in fuse.inputs().iter() {
                    exprs.push(self._lower(input, false));
                }
                if fuse.inputs().len() == 1 && fuse.inputs()[0].is_reduce() {
                    exprs.push(fuse.inputs()[0].to_reduce().identity().clone());
                    let call: PrimeExpr = fuse.func().call_common(exprs).into();
                    let mut contains = true;
                    if let Some(saved) = self.cnts.get_mut(&fuse.id()) {
                        if !self.unique_exprs.contains(&call) {
                            contains = false;
                            self.unique_exprs.insert(call.clone());
                            *saved += 1;
                        }
                    } else {
                        self.cnts.insert(fuse.id(), 0);
                        self.unique_exprs.insert(call.clone());
                        contains = false;
                    }
                    let var = Variable::new(format!("%{}_{}", fuse.id(), self.cnts[&fuse.id()]));
                    if !contains {
                        self.ordered_exprs.push((var.clone().into(), call.into()));
                    }
                    return var.into();
                } else {
                    let call = fuse.func().call_common(exprs);
                    call.into()
                }
            }
        }
    }
}

impl Display for HlirLower {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for (k, v) in self.ordered_exprs.iter() {
            write!(f, "{} = {}\n", k, v)?;
        }
        Ok(())
    }
}
