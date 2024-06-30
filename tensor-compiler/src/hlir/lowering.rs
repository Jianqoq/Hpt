use std::fmt::Display;

use hashbrown::{ HashMap, HashSet };
use tensor_common::shape::Shape;
use tensor_types::dtype::Dtype;

use crate::{
    edges::Edges,
    halide::{
        assign_stmt::AssignStmt,
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
    ordered_exprs: Vec<(String, Option<PrimeExpr>, PrimeExpr)>,
    /// this unique exprs is the unique exprs in the computation
    unique_exprs: HashSet<PrimeExpr>,
    /// in compilation, each node could have multiple different compute
    /// so we need to count the number of each node so that we can generate unique name for each expr
    cnts: HashMap<usize, usize>,
    /// a stack to get the for loop variables correctly
    var_stack: Vec<Variable>,
    /// for loop index variable
    loop_indexes: Vec<Vec<(Variable, i64)>>,
    /// for loop range
    loop_ranges: Vec<Vec<(i64, i64, i64)>>, // (start, end, step)
    lowered_fors: Vec<(String, Stmt)>,
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
            let variable_name = Variable::from(format!("ptr{}", tensor.id()));
            let name = variable_name.name().to_string();
            for input in self.common_depends.iter() {
                self.loop_dependencies
                    .entry(name.clone())
                    .or_insert(HashSet::new())
                    .insert(input.clone());
            }
            let common_var_stack = self.var_stack
                .iter()
                .zip(self.common_shape.iter())
                .map(|(x, y)| (x.clone(), *y))
                .collect::<Vec<_>>();
            self.loop_indexes.push(common_var_stack);
            self.ordered_exprs.push((variable_name.name().to_string(), None, expr.clone()));
        }

        self._build_nested_loop();
    }

    fn _lower(&mut self, output: &Tensor, root: bool) -> PrimeExpr {
        match output {
            Tensor::Base(base) => {
                let mut strides = base.layout().strides().to_vec();
                let reduced_strides = base.reduced_strides().to_vec();
                strides.extend(reduced_strides.iter());
                assert!(self.var_stack.len() == strides.len());
                let variable_name = Variable::from(format!("ptr{}", base.id()));
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
                if root {
                    for i in 0..fuse.ndim() {
                        self.var_stack.push(Variable::new(format!("i{}", i)));
                    }
                }
                let mut exprs = vec![];
                for input in fuse.inputs().iter() {
                    if root {
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
                    let call: PrimeExpr = fuse.func().call_common(exprs.clone()).into();
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
                    let var = Variable::new(format!("{}_{}", fuse.id(), self.cnts[&fuse.id()]));
                    if !contains {
                        // the identity will be replaced with the variable
                        exprs[1] = var.clone().into();
                        let call = fuse.func().call_common(exprs);
                        self.ordered_exprs.push((
                            var.name.to_string(),
                            Some(reduce.identity().clone()),
                            call.into(),
                        ));
                        let mut vec = vec![];
                        for (var, dim) in reduce.reduce_vars().iter().zip(reduce.shape().iter()) {
                            vec.push((var.clone(), *dim));
                        }
                        self.loop_indexes.push(vec);
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
        let mut loop_indexes_map = HashMap::new();
        for (loop_index, (name, _, _)) in self.loop_indexes.iter().zip(self.ordered_exprs.iter()) {
            loop_indexes_map.insert(name, loop_index);
        }
        let mut sorted_edges = HashMap::new();
        for (name, depends) in self.loop_dependencies.iter() {
            let mut ordered_set = depends
                .iter()
                .map(|x| x)
                .collect::<Vec<_>>();

            ordered_set.sort_by_key(|k|
                self.ordered_exprs
                    .iter()
                    .map(|(x, _, _)| x)
                    .position(|x| x == *k)
                    .expect(&format!("{} not found in ordered_exprs", k))
            );
            sorted_edges.insert(name, ordered_set);
        }
        let mut loop_generated = HashMap::<&String, Stmt>::new();
        for ((name, identity, body), indexes) in self.ordered_exprs
            .iter()
            .zip(self.loop_indexes.iter()) {
            let shape: Shape = indexes
                .iter()
                .map(|(_, dim)| *dim)
                .collect::<Vec<_>>()
                .into();
            let vars = indexes
                .iter()
                .map(|(var, _)| var.clone())
                .collect::<Vec<_>>();

            let dependencies = &sorted_edges[name];

            let mut seq = vec![];

            let tmp_val = Variable::make(name);

            for dep in dependencies.iter() {
                if let Some(stmt) = loop_generated.get(dep) {
                    seq.push(stmt.clone());
                } else {
                    panic!("{} not found in loop_generated", dep);
                }
            }
            if identity.is_some() {
                seq.push(AssignStmt::make(&tmp_val, body.clone()).into());
            } else {
                seq.push(LetStmt::make(&tmp_val, body.clone()).into());
            }
            let mut stmt = build_nested_for(&vars, &shape, Seq::make(seq));
            if let Some(init) = identity {
                stmt = Seq::make(vec![LetStmt::make(&tmp_val, init.clone()).into(), stmt]).into();
            }
            self.lowered_fors.push((name.clone(), stmt.clone()));
            loop_generated.insert(name, stmt);
        }
    }

    pub fn print_lowered_fors(&self) {
        if let Some(last) = self.lowered_fors.last() {
            IRPrinter.print_stmt(last.1.clone());
        } else {
            println!("No lowered for loop found");
        }
    }
    pub fn loop_dependencies(&self) -> &Edges<String> {
        &self.loop_dependencies
    }
}

impl Display for HlirLower {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for (loop_idx, (name, _, compute)) in self.loop_indexes
            .iter()
            .zip(self.ordered_exprs.iter()) {
            write!(
                f,
                "{} = [{}], {}\n",
                name,
                loop_idx
                    .iter()
                    .map(|(var, dim)| format!("({}:{})", var.name(), dim))
                    .collect::<Vec<String>>()
                    .join(", "),
                compute
            )?;
        }
        Ok(())
    }
}
