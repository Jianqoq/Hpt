use std::{ panic::Location, sync::Arc };

use tensor_types::dtype::Dtype;

use crate::{
    halide::{
        exprs::Load,
        let_stmt::LetStmt,
        prime_expr::PrimeExpr,
        stmt::Stmt,
        store_stmt::StoreStmt,
        substitute::subsititue_var::SubstituteVar,
        utils::{ dtype_one, exp, neg },
        variable::Variable,
    },
    iter_var::IterVar,
    te::{
        context::Context,
        stages::{ Body, Stage },
        strides_cal_helper::elementwise_strides_cal,
        tensor::{ StridesCal, Tensor },
    },
};

pub fn common_sigmoid(
    is_output: bool,
    inputs: &Vec<Body>,
    shape: &Vec<PrimeExpr>,
    res_dtype: Dtype,
    output_id: usize
) -> Body {
    let func = |stage_out_id: usize| {
        let var: PrimeExpr = Variable::make(&format!("%{}_val", stage_out_id)).into();
        let one = dtype_one(res_dtype);
        one.clone() / (one + exp(neg(var)))
    };
    if is_output {
        common_sigmoid_helper(
            inputs,
            shape,
            output_id,
            res_dtype,
            |dims: &Vec<IterVar>, stage_out_id: usize, stage: &Stage|
                Body::Stmt(
                    StoreStmt::make(
                        format!("%{}", output_id),
                        stage.begins.clone(),
                        stage.axes.clone(),
                        stage.steps.clone(),
                        (0..dims.len())
                            .map(|x| { Load::make(&format!("%{}.s", output_id), x).into() })
                            .collect::<Vec<PrimeExpr>>(),
                        func(stage_out_id)
                    ).into()
                )
        )
    } else {
        common_sigmoid_helper(inputs, shape, output_id, res_dtype, |_, stage_out_id: usize, _|
            Body::Stmt(
                Stmt::LetStmt(
                    LetStmt::make(
                        &Variable::make(&format!("%{}_val", output_id)),
                        func(stage_out_id),
                        false,
                        Stmt::None
                    )
                ).into()
            )
        )
    }
}

pub fn common_sigmoid_helper<F>(
    inputs: &Vec<Body>,
    shape: &Vec<PrimeExpr>,
    output_id: usize,
    res_dtype: Dtype,
    post: F
) -> Body
    where F: Fn(&Vec<IterVar>, usize, &Stage) -> Body
{
    match &inputs[0] {
        Body::Stage(stage) => {
            let mut stage = stage.clone();
            let dims = (0..shape.len())
                .map(|x| IterVar::new(0i64, shape[x].clone(), 1i64, &format!("ax{}", x)))
                .collect::<Vec<IterVar>>();

            let mut subs_var = SubstituteVar::new();
            subs_var.add_replacement(
                Variable::new(format!("%{}.s", stage.out_id)),
                Variable::new(format!("%{}.s", output_id))
            );
            stage.bodys.push(post(&dims, stage.out_id, &stage));

            for i in stage.bodys.iter_mut() {
                i.accept_mutate(&mut subs_var);
            }
            let stage = Stage {
                dims,
                bodys: stage.bodys.clone(),
                id: output_id,
                out_id: stage.out_id,
                dtype: res_dtype,
                begins: stage.begins.clone(),
                steps: stage.steps.clone(),
                axes: stage.axes.clone(),
            };
            Body::Stage(stage)
        }
        _ => panic!("input is not a stage"),
    }
}

impl Context {
    #[track_caller]
    pub fn sigmoid(&mut self, a: &Tensor) -> Tensor {
        let id = self.id.borrow().clone();
        *self.id.borrow_mut() += 1;
        let shape = a.shape.clone();
        let res_dtype = a.dtype;
        let ret = Tensor {
            shape: a.shape.clone(),
            inputs: Arc::new(vec![a.id]),
            span: Location::caller(),
            id,
            dtype: res_dtype,
            strides_cal: Arc::new(move |prev_fn: Vec<StridesCal>| {
                elementwise_strides_cal(prev_fn[0].clone())
            }),
            body_gen: Arc::new(move |inputs: Vec<Body>, is_output: bool, id: usize| {
                common_sigmoid(is_output, &inputs, &shape, res_dtype, id)
            }),
        };
        self.nodes.borrow_mut().insert(id, ret.clone());
        ret
    }
}
