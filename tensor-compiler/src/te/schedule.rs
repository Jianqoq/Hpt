use std::{ collections::{ HashMap, HashSet }, fmt::{ self, Display, Formatter }, sync::Arc };

use tensor_types::dtype::Dtype;

use crate::{
    halide::{
        exprs::Load,
        let_stmt::LetStmt,
        module::{ Function, FunctionType },
        primitive_type::{ PrimitiveType, Ptr },
        printer::IRPrinter,
        seq_stmt::Seq,
        stmt::Stmt,
        traits::AccepterMutate,
        variable::Variable,
    },
    te::strides_visitor::StridesVisitor,
};

use super::{ hstrides::HStrides, rc_mut::RcMut, stages::Body, tensor::Tensor };

pub struct Schedule {
    pub(crate) qa: HashMap<usize, (Body, bool)>,
    pub(crate) nodes: RcMut<HashMap<usize, Tensor>>,
    pub(crate) strides_cal: Arc<dyn Fn(&HashMap<String, i64>) -> Vec<HStrides>>,
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
        let strides_loads = gen_strides_loads(strides_visitor.cnt);
        let ptr_loads = gen_ptr_loads(self);
        let shape_var_loads = gen_shape_var_loads(self);
        let mut empty_fn = empty_fn();

        let mut body = [strides_loads, ptr_loads, shape_var_loads]
            .into_iter()
            .flatten()
            .collect::<Vec<Stmt>>();
        body.push(new_fn_body);

        empty_fn.body = Stmt::Seq(Seq::make(body));

        empty_fn
    }

    pub fn cal_strides(&self, var_map: &HashMap<String, i64>) -> Vec<HStrides> {
        (self.strides_cal)(var_map)
    }
}

fn replace_strides(stmt: &Stmt, strides_visitor: &mut StridesVisitor) -> Stmt {
    stmt.accept_mutate(strides_visitor);
    strides_visitor.stmt.clone()
}

fn gen_strides_loads(strides_num: i64) -> Vec<Stmt> {
    (0..strides_num)
        .map(|idx| {
            let let_stmt = LetStmt::make(
                &Variable::make(&format!("strides{}", idx)),
                Load::make("strides_vec", idx),
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
            input_vec.push(node.id);
        }
    }
    input_vec.sort();
    input_vec
        .iter()
        .enumerate()
        .map(|(idx, x)| {
            Stmt::LetStmt(
                LetStmt::make(
                    &Variable::make(&format!("%{}", x)),
                    Load::make("data_vec", idx),
                    Stmt::None
                )
            )
        })
        .collect::<Vec<Stmt>>()
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
            Stmt::LetStmt(LetStmt::make(x, Load::make("shape_vars", idx), Stmt::None))
        })
        .collect::<Vec<Stmt>>()
}

fn empty_fn() -> Function {
    Function {
        ty: FunctionType {
            ret_ty: PrimitiveType::Void,
            args: Arc::new(
                vec![
                    (
                        format!("strides_vec"),
                        PrimitiveType::Ptr(Ptr {
                            inner: Arc::new(
                                PrimitiveType::Ptr(Ptr {
                                    inner: PrimitiveType::Dtype(Dtype::I64).into(),
                                })
                            ),
                        }),
                    ),
                    (
                        format!("data_vec"),
                        PrimitiveType::Ptr(Ptr {
                            inner: Arc::new(
                                PrimitiveType::Ptr(Ptr { inner: PrimitiveType::Void.into() })
                            ),
                        }),
                    ),
                    (
                        format!("offset_vec"),
                        PrimitiveType::Ptr(Ptr {
                            inner: Arc::new(PrimitiveType::Dtype(Dtype::I64)),
                        }),
                    ),
                    (
                        format!("shape_vars"),
                        PrimitiveType::Ptr(Ptr {
                            inner: Arc::new(PrimitiveType::Dtype(Dtype::I64)),
                        }),
                    ),
                    (format!("thread_idx"), PrimitiveType::Dtype(Dtype::I64))
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
