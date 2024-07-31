use std::{ collections::{ HashMap, HashSet }, fmt::{ self, Display, Formatter }, sync::Arc };

use tensor_types::dtype::Dtype;

use crate::{
    edges::Edges,
    halide::{
        exprs::{ BitCast, Load },
        let_stmt::LetStmt,
        module::{ Function, FunctionType },
        prime_expr::PrimeExpr,
        primitive_type::{ PrimitiveType, Ptr },
        printer::IRPrinter,
        return_stmt::ReturnStmt,
        seq_stmt::Seq,
        stmt::Stmt,
        traits::AccepterMutate,
        variable::Variable,
    },
    te::strides_visitor::StridesVisitor,
};

use super::{
    hstrides::HStrides,
    rc_mut::RcMut,
    stages::Body,
    strides_visitor::StridesStoreVisitor,
    tensor::Tensor,
};

pub struct Schedule {
    pub(crate) qa: HashMap<usize, (Body, bool)>,
    pub(crate) nodes: RcMut<HashMap<usize, Tensor>>,
    pub(crate) strides_cal: Arc<dyn Fn(&HashMap<Arc<String>, i64>) -> Vec<HStrides>>,
}

impl Schedule {
    pub fn to_halide(&self) -> Vec<Stmt> {
        let mut ret = Vec::new();
        for (body, is_output) in self.qa.values() {
            match body {
                Body::Stage(stage) => {
                    if *is_output {
                        ret.push(stage.to_halide(&self.qa));
                    }
                }
                _ => {}
            }
        }
        ret
    }

    pub fn to_function(&self) -> Function {
        let fn_body = Stmt::Seq(Seq::make(self.to_halide()));

        let mut strides_visitor = StridesVisitor::new();
        let new_fn_body = replace_strides(&fn_body, &mut strides_visitor);
        let mut strides_store_visitor = StridesStoreVisitor::new();
        let new_fn_body = replace_strides_store(&new_fn_body, &mut strides_store_visitor);
        let strides_loads = gen_strides_loads(strides_visitor.cnt);
        let ptr_loads = gen_ptr_loads(self);
        let ptr_stores = gen_ptr_stores(self);
        let strides_outs = gen_strides_outs(ptr_stores.len() as i64);
        let shape_var_loads = gen_shape_var_loads(self);
        let mut empty_fn = empty_fn();

        let mut body = [strides_loads, ptr_loads, ptr_stores, strides_outs, shape_var_loads]
            .into_iter()
            .flatten()
            .collect::<Vec<Stmt>>();
        body.push(new_fn_body);
        body.push(Stmt::Return(ReturnStmt::make(vec![])));

        empty_fn.body = Stmt::Seq(Seq::make(body));

        empty_fn
    }

    pub fn cal_strides(&self, var_map: &HashMap<Arc<String>, i64>) -> Vec<HStrides> {
        (self.strides_cal)(var_map)
    }
    pub fn inputs_order(&self) -> Vec<usize> {
        inps_order(self)
    }
    pub fn outputs_order(&self) -> Vec<usize> {
        outs_order(self)
    }
    pub fn inputs_dtype(&self) -> Vec<Dtype> {
        self.inputs_order()
            .iter()
            .map(|x| self.nodes.borrow().get(x).unwrap().dtype)
            .collect()
    }
    pub fn outputs_dtype(&self) -> Vec<Dtype> {
        self.outputs_order()
            .iter()
            .map(|x| self.nodes.borrow().get(x).unwrap().dtype)
            .collect()
    }
    pub fn outs_shape(&self) -> Vec<Arc<Vec<PrimeExpr>>> {
        self.outputs_order()
            .iter()
            .map(|x| {
                let shape = self.nodes.borrow().get(x).unwrap().shape.clone();
                if shape.len() == 0 {
                    vec![1i64.into()].into()
                } else {
                    shape
                }
            })
            .collect()
    }
    pub fn inps_shape(&self) -> Vec<Arc<Vec<PrimeExpr>>> {
        self.inputs_order()
            .iter()
            .map(|x| {
                let shape = self.nodes.borrow().get(x).unwrap().shape.clone();
                if shape.len() == 0 {
                    vec![1i64.into()].into()
                } else {
                    shape
                }
            })
            .collect()
    }
    pub fn vars_order(&self) -> Vec<Arc<String>> {
        vars_order(self)
    }
    pub fn inputs(&self) -> HashSet<usize> {
        self.nodes
            .borrow()
            .iter()
            .filter(|(_, x)| x.inputs.len() == 0)
            .map(|(x, _)| *x)
            .collect()
    }

    pub fn outputs(&self) -> HashSet<usize> {
        let mut edges = HashMap::new();
        for node in self.nodes.borrow().values() {
            if edges.contains_key(&node.id) {
                continue;
            } else {
                edges.insert(node.id, HashSet::from_iter(node.inputs.iter().map(|x| *x)));
            }
        }
        let mut edge = Edges::new();
        edge.set_inner(edges);
        let mut inverted = edge.invert();
        for i in edge.keys() {
            inverted.entry(*i).or_insert(HashSet::new());
        }
        inverted
            .iter()
            .filter(|(_, outputs)| outputs.len() == 0)
            .map(|x| *x.0)
            .collect::<HashSet<usize>>()
    }
}

fn replace_strides(stmt: &Stmt, strides_visitor: &mut StridesVisitor) -> Stmt {
    stmt.accept_mutate(strides_visitor);
    strides_visitor.stmt.clone()
}

fn replace_strides_store(stmt: &Stmt, strides_store_visitor: &mut StridesStoreVisitor) -> Stmt {
    stmt.accept_mutate(strides_store_visitor);
    strides_store_visitor.stmt.clone()
}

fn gen_strides_loads(strides_num: i64) -> Vec<Stmt> {
    (0..strides_num)
        .map(|idx| {
            let let_stmt = LetStmt::make(
                &Variable::make(&format!("istrides{}", idx)),
                Load::make("istrides_vec", idx),
                false,
                Stmt::None
            );
            Stmt::LetStmt(let_stmt)
        })
        .collect::<Vec<Stmt>>()
}

fn gen_strides_outs(strides_num: i64) -> Vec<Stmt> {
    (0..strides_num)
        .map(|idx| {
            let let_stmt = LetStmt::make(
                &Variable::make(&format!("ostrides{}", idx)),
                Load::make("ostrides_vec", idx),
                false,
                Stmt::None
            );
            Stmt::LetStmt(let_stmt)
        })
        .collect::<Vec<Stmt>>()
}

fn gen_ptr_loads(schedule: &Schedule) -> Vec<Stmt> {
    let mut input_vec = vec![];
    for node in schedule.nodes.borrow().values() {
        if node.inputs.len() == 0 {
            input_vec.push((node.id, node.dtype));
        }
    }
    input_vec.sort();
    input_vec
        .iter()
        .enumerate()
        .map(|(idx, (x, dtype))| {
            Stmt::LetStmt(
                LetStmt::make(
                    &Variable::make(&format!("%{}", x)),
                    BitCast::make(
                        Load::make("data_vec", idx),
                        PrimitiveType::Ptr(Ptr {
                            inner: Arc::new(PrimitiveType::Dtype(*dtype)),
                        })
                    ),
                    false,
                    Stmt::None
                )
            )
        })
        .collect::<Vec<Stmt>>()
}

fn inps_order(schedule: &Schedule) -> Vec<usize> {
    let mut input_vec = vec![];
    for node in schedule.nodes.borrow().values() {
        if node.inputs.len() == 0 {
            input_vec.push(node.id);
        }
    }
    input_vec.sort();
    input_vec
}

fn gen_ptr_stores(schedule: &Schedule) -> Vec<Stmt> {
    let mut edges = HashMap::new();
    for node in schedule.nodes.borrow().values() {
        if edges.contains_key(&(node.id, node.dtype)) {
            continue;
        } else {
            edges.insert(
                (node.id, node.dtype),
                HashSet::from_iter(
                    node.inputs.iter().map(|x| (*x, schedule.nodes.borrow().get(x).unwrap().dtype))
                )
            );
        }
    }
    let mut edge = Edges::new();
    edge.set_inner(edges);
    let mut inverted = edge.invert();
    for i in edge.keys() {
        inverted.entry(*i).or_insert(HashSet::new());
    }
    let mut outputs = inverted
        .iter()
        .filter(|(_, outputs)| outputs.len() == 0)
        .map(|x| *x.0)
        .collect::<Vec<(usize, Dtype)>>();
    outputs.sort();
    outputs
        .iter()
        .enumerate()
        .map(|(idx, (x, dtype))| {
            Stmt::LetStmt(
                LetStmt::make(
                    &Variable::make(&format!("%{}", x)),
                    BitCast::make(
                        Load::make("output_vec", idx),
                        PrimitiveType::Ptr(Ptr {
                            inner: Arc::new(PrimitiveType::Dtype(*dtype)),
                        })
                    ),
                    false,
                    Stmt::None
                )
            )
        })
        .collect::<Vec<Stmt>>()
}

fn outs_order(schedule: &Schedule) -> Vec<usize> {
    let mut edges = HashMap::new();
    for node in schedule.nodes.borrow().values() {
        if edges.contains_key(&node.id) {
            continue;
        } else {
            edges.insert(node.id, HashSet::from_iter(node.inputs.iter().map(|x| *x)));
        }
    }
    let mut edge = Edges::new();
    edge.set_inner(edges);
    let mut inverted = edge.invert();
    for i in edge.keys() {
        inverted.entry(*i).or_insert(HashSet::new());
    }
    let mut outputs = inverted
        .iter()
        .filter(|(_, outputs)| outputs.len() == 0)
        .map(|x| *x.0)
        .collect::<Vec<usize>>();
    outputs.sort();
    outputs
}

fn gen_shape_var_loads(schedule: &Schedule) -> Vec<Stmt> {
    let mut vars = vec![];
    let mut visited = HashSet::new();
    for node in schedule.nodes.borrow().values() {
        for i in node.shape.iter() {
            if let Some(var) = i.to_variable() {
                if visited.contains(var) {
                    continue;
                } else {
                    vars.push(var.clone());
                    visited.insert(var);
                }
            }
        }
    }
    vars.sort();
    vars.iter()
        .enumerate()
        .map(|(idx, x)| {
            Stmt::LetStmt(LetStmt::make(x, Load::make("shape_vars", idx), false, Stmt::None))
        })
        .collect::<Vec<Stmt>>()
}

fn vars_order(schedule: &Schedule) -> Vec<Arc<String>> {
    let mut vars = vec![];
    let mut visited = HashSet::new();
    for node in schedule.nodes.borrow().values() {
        for i in node.shape.iter() {
            if let Some(var) = i.to_variable() {
                if visited.contains(var) {
                    continue;
                } else {
                    vars.push(var.name.clone());
                    visited.insert(var);
                }
            }
        }
    }
    vars.sort();
    vars
}

fn empty_fn() -> Function {
    Function {
        ty: FunctionType {
            ret_ty: PrimitiveType::Void,
            args: Arc::new(
                vec![
                    (
                        "istrides_vec".to_string(),
                        PrimitiveType::Ptr(Ptr {
                            inner: Arc::new(
                                PrimitiveType::Ptr(Ptr {
                                    inner: PrimitiveType::Dtype(Dtype::I64).into(),
                                })
                            ),
                        }),
                    ),
                    (
                        "ostrides_vec".to_string(),
                        PrimitiveType::Ptr(Ptr {
                            inner: Arc::new(
                                PrimitiveType::Ptr(Ptr {
                                    inner: PrimitiveType::Dtype(Dtype::I64).into(),
                                })
                            ),
                        }),
                    ),
                    (
                        "data_vec".to_string(),
                        PrimitiveType::Ptr(Ptr {
                            inner: Arc::new(
                                PrimitiveType::Ptr(Ptr { inner: PrimitiveType::Void.into() })
                            ),
                        }),
                    ),
                    (
                        "output_vec".to_string(),
                        PrimitiveType::Ptr(Ptr {
                            inner: Arc::new(
                                PrimitiveType::Ptr(Ptr { inner: PrimitiveType::Void.into() })
                            ),
                        }),
                    ),
                    (
                        "offset_vec".to_string(),
                        PrimitiveType::Ptr(Ptr {
                            inner: Arc::new(PrimitiveType::Dtype(Dtype::I64)),
                        }),
                    ),
                    (
                        "shape_vars".to_string(),
                        PrimitiveType::Ptr(Ptr {
                            inner: Arc::new(PrimitiveType::Dtype(Dtype::I64)),
                        }),
                    ),
                    ("thread_idx".to_string(), PrimitiveType::Dtype(Dtype::I64))
                ]
            ),
        },
        name: Arc::new("kernel".to_string()),
        body: Stmt::None,
    }
}

impl Display for Schedule {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            self
                .to_halide()
                .iter()
                .map(|x| IRPrinter.print_stmt_str(x))
                .collect::<Vec<String>>()
                .join("\n")
        )
    }
}
