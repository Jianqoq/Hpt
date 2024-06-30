use std::fmt::Display;

use hashbrown::{ HashMap, HashSet };
use tensor_common::shape::Shape;
use tensor_types::dtype::Dtype;

use crate::{
    edges::Edges,
    halide::{
        exprs::{ Int, Load },
        let_stmt::LetStmt,
        loop_utils::build_nested::build_nested_for,
        prime_expr::PrimeExpr,
        printer::IRPrinter,
        seq_stmt::Seq,
        stmt::Stmt,
        variable::Variable,
    },
};

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
    /// temp var stack
    temp_var_stack: Vec<Variable>,
    /// for loop index variable
    loop_indexes: Vec<Vec<Variable>>,
    /// for loop range
    loop_ranges: Vec<Vec<(i64, i64, i64)>>, // (start, end, step)
    lowered_fors: Vec<(PrimeExpr, Stmt)>,
    loop_dependencies: Edges<String>,
    temp_loop_ids: HashSet<String>,
    common_depends: Vec<String>,
    common_shape: Shape,
}

impl HlirLower {
    pub fn new() -> Self {
        Self {
            ordered_exprs: vec![],
            unique_exprs: HashSet::new(),
            cnts: HashMap::new(),
            var_stack: vec![],
            loop_indexes: vec![],
            temp_var_stack: vec![],
            loop_ranges: vec![],
            lowered_fors: vec![],
            common_shape: Shape::new(vec![]),
            loop_dependencies: Edges::new(),
            temp_loop_ids: HashSet::new(),
            common_depends: vec![],
        }
    }

    pub fn lower<T: Into<Tensor>>(&mut self, output: T) {
        let tensor: &Tensor = &output.into();
        self.common_shape = tensor.shape().clone();
        let expr = self._lower(tensor, true);
        if !self.unique_exprs.contains(&expr) {
            self.unique_exprs.insert(expr.clone());
            let variable_name = Variable::from(format!("%{}", tensor.id()));
            let name = variable_name.name().to_string();
            for input in self.common_depends.iter() {
                self.loop_dependencies
                    .entry(name.clone())
                    .or_insert(HashSet::new())
                    .insert(input.clone());
            }
            self.ordered_exprs.push((variable_name.into(), expr.clone()));
            if tensor.is_reduce() {
                self.loop_indexes.push(self.temp_var_stack.clone());
                let reduce_shape = tensor.to_reduce().shape().clone();
                let mut vec = vec![];
                for i in 0..reduce_shape.len() {
                    vec.push((0, reduce_shape[i] as i64, 1));
                }
                self.loop_ranges.push(vec);
            }
        }
        self._build_nested_loop();
    }

    fn _lower(&mut self, output: &Tensor, push_vars: bool) -> PrimeExpr {
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
                    self.temp_var_stack.push(var.clone());
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
                    if push_vars {
                        self.temp_loop_ids.clear();
                        exprs.push(self._lower(input, false));
                        self.common_depends.extend(self.temp_loop_ids.iter().cloned());
                    } else {
                        exprs.push(self._lower(input, false));
                    }
                }
                if fuse.inputs().len() == 1 && fuse.inputs()[0].is_reduce() {
                    let reduce = fuse.inputs()[0].to_reduce();
                    exprs.push(reduce.identity().clone());
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
                        self.loop_indexes.push(self.temp_var_stack.clone());
                        self.loop_dependencies.insert(
                            var.name().to_string(),
                            self.temp_loop_ids.clone()
                        );
                        self.temp_loop_ids.clear();
                        let reduce_shape = reduce.shape().clone();
                        let mut vec = vec![];
                        for i in 0..reduce_shape.len() {
                            vec.push((0, reduce_shape[i] as i64, 1));
                        }
                        self.loop_ranges.push(vec);
                    }
                    for _ in reduce.reduce_vars().iter() {
                        self.temp_var_stack.pop();
                    }
                    self.temp_loop_ids.insert(var.name().to_string());
                    return var.into();
                } else {
                    let call = fuse.func().call_common(exprs);
                    call.into()
                }
            }
        }
    }

    fn _build_nested_loop(&mut self) {
        for (name, body) in self.ordered_exprs.iter() {
        }
    }

    pub fn print_lowered_fors(&self) {
        for (name, stmt) in self.lowered_fors.iter() {
            print!("{} =\n", name);
            IRPrinter.print_stmt(stmt);
        }
    }
    pub fn loop_dependencies(&self) -> &Edges<String> {
        &self.loop_dependencies
    }
}

impl Display for HlirLower {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for (loop_idx, (name, compute)) in self.loop_indexes.iter().zip(self.ordered_exprs.iter()) {
            write!(
                f,
                "{} = [{}], {}\n",
                name,
                loop_idx
                    .iter()
                    .map(|x| x.name())
                    .collect::<Vec<&str>>()
                    .join(", "),
                compute
            )?;
        }
        if self.loop_indexes.len() < self.ordered_exprs.len() {
            let (name, compute) = self.ordered_exprs.last().unwrap();
            write!(
                f,
                "{} = [{}], {}\n",
                name,
                self.var_stack
                    .iter()
                    .map(|x| x.name())
                    .collect::<Vec<&str>>()
                    .join(", "),
                compute
            )?;
        }
        Ok(())
    }
}
