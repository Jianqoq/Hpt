use std::{ collections::HashMap, panic::Location, sync::Arc };

use crate::{
    halide::{
        alloca_stmt::AllocaStmt,
        exprs::{ BitAnd, Ge, Load, Lt },
        let_stmt::LetStmt,
        passes::const_fold::ConstFold,
        prime_expr::PrimeExpr,
        primitive_type::PrimitiveType,
        stmt::Stmt,
        store_stmt::StoreStmt,
        tensor_load::TensorLoad,
        utils::{ all, bitand, dtype_zero, floor },
        variable::Variable,
    },
    iter_var::IterVar,
    te::{
        context::Context,
        stages::{ Body, If, ReduceStage, Stage },
        tensor::{ StridesCal, Tensor },
    },
    to_prim_expr::ToPrimeExpr,
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AutoPad {
    #[default]
    Notset,
    SameUpper,
    SameLower,
    Valid,
}

impl Context {
    /// ### Convolution
    ///
    /// input: has size `(N x C x H x W)`, where `N` is the batch size, `C` is the number of channels, and `H` and `W` are the height and width.
    /// Note that this is for the 2D image. Otherwise the size is `(N x C x D1 x D2 … x Dn)`
    ///
    /// weight: The weight tensor that will be used in the convolutions;
    /// has size `(M x C/group x kH x kW)`, where `C` is the number of channels,
    /// and `kH` and `kW` are the height and width of the kernel, and `M` is the number of feature maps.
    ///
    /// bias: Optional `1D` bias to be added to the convolution, has size of `M`.
    #[track_caller]
    pub fn conv(
        &mut self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        kernel_shape: Option<&[&dyn ToPrimeExpr]>,
        pads: Option<&[(&dyn ToPrimeExpr, &dyn ToPrimeExpr)]>,
        steps: Option<&[&dyn ToPrimeExpr]>,
        auto_pad: Option<AutoPad>,
        dilations: Option<&[i64]>,
        group: Option<i64>
    ) -> Tensor {
        let id = self.id.borrow().clone();
        *self.id.borrow_mut() += 1;
        let caller = Location::caller();
        let auto_pad = auto_pad.unwrap_or(AutoPad::Notset);
        let tmp: Vec<PrimeExpr> = if let Some(tmp) = kernel_shape {
            tmp.iter()
                .map(|x| x.to_prime_expr())
                .collect()
        } else {
            weight.shape.iter().skip(2).cloned().collect()
        };
        let pads = if let Some(pads) = pads {
            pads.iter()
                .map(|(x, y)| (x.to_prime_expr(), y.to_prime_expr()))
                .collect()
        } else {
            vec![(0i64.into(), 0i64.into()); tmp.len()]
        };
        let steps = if let Some(steps) = steps {
            steps
                .iter()
                .map(|x| x.to_prime_expr())
                .collect()
        } else {
            vec![1i64.into(); tmp.len()]
        };
        let dilations = if let Some(dilations) = dilations {
            dilations.iter().cloned().collect()
        } else {
            vec![1; tmp.len()]
        };

        let bias_id = bias.map(|x| x.id);
        let mut outer_dims = vec![];
        outer_dims.push(input.shape[0].clone());
        outer_dims.push(weight.shape[0].clone());
        let mut out_dims = vec![];
        input.shape
            .iter()
            .skip(2)
            .zip(weight.shape.iter().skip(2))
            .enumerate()
            .for_each(|(idx, (x, y))| {
                let i = x.clone();
                let (p_begin, p_end) = pads[idx].clone();
                let d: PrimeExpr = dilations[idx].into();
                let s = steps[idx].clone();
                let k = y.clone();
                let o = floor((i + p_begin + p_end - d * (k - 1i64) - 1i64) / s + 1i64);
                out_dims.push(ConstFold::new().const_fold(o));
            });
        outer_dims.append(&mut out_dims);
        let outer_dims = Arc::new(outer_dims);
        let shape = outer_dims.clone();
        let groups: PrimeExpr = group.unwrap_or(1i64).into();
        let out_channels_per_group = &weight.shape[0] / &groups;
        let body = move |inputs: Vec<Body>, is_output: bool, id: usize| {
            match (&inputs[0], &inputs[1]) {
                (Body::Stage(input), Body::Stage(kernel)) => {
                    let mut kernel_dims = Vec::with_capacity(kernel.dims[2..].len());
                    for (idx, i) in kernel.dims.iter().enumerate().skip(2) {
                        let mut i = i.clone();
                        i.set_var(Variable::new(format!("{}red{}", kernel.id, idx)));
                        kernel_dims.push(i);
                    }
                    let sum_ptr = Variable::new(format!("%{}_val_ptr", id));
                    let add_bias = if let Some(bias) = bias_id {
                        let load_sum: PrimeExpr = Load::make(&sum_ptr, 0i64).into();
                        let bias_val: PrimeExpr = Variable::new(format!("%{}_val", bias)).into();
                        let add = load_sum + bias_val;
                        Body::Stmt(Stmt::StoreStmt(StoreStmt::make(&sum_ptr, 0i64, add)).into())
                    } else {
                        Body::Stmt(Stmt::None)
                    };
                    let mut dims = outer_dims.as_ref().clone();
                    dims[1] = groups.clone();
                    dims.insert(2, out_channels_per_group.clone());
                    let dims = dims
                        .into_iter()
                        .enumerate()
                        .map(|(idx, x)| {
                            IterVar::new(0i64, x, 1i64, Variable::new(format!("ax{}", idx)))
                        })
                        .collect::<Vec<IterVar>>();
                    let inp_begins = dims
                        .iter()
                        .skip(3)
                        .map(|x| x.var().to_prime_expr())
                        .collect::<Vec<_>>();
                    println!("{:?}", inp_begins);
                    let kernel_begins = kernel_dims
                        .iter()
                        .map(|x| x.var().to_prime_expr())
                        .collect::<Vec<_>>();
                    let begins = inp_begins
                        .iter()
                        .zip(kernel_begins.iter())
                        .zip(steps.iter())
                        .zip(dilations.iter())
                        .map(|(((x, y), z), w)| {
                            x.clone() * z.to_prime_expr() + y.clone() * w.to_prime_expr()
                        })
                        .collect::<Vec<_>>();
                    let mut lets_ = begins
                        .iter()
                        .enumerate()
                        .map(|(i, x)|
                            Body::Stmt(
                                Stmt::LetStmt(
                                    LetStmt::make(
                                        &Variable::new(format!("inp{}", i)),
                                        x,
                                        false,
                                        Stmt::None
                                    )
                                )
                            )
                        )
                        .collect::<Vec<Body>>();
                    lets_.push(
                        Body::Stmt(
                            Stmt::LetStmt(
                                LetStmt::make(
                                    &Variable::new(format!("%{}_val", id)),
                                    Load::make(&sum_ptr, 0i64),
                                    false,
                                    Stmt::None
                                )
                            )
                        )
                    );
                    let cmps = pads
                        .iter()
                        .enumerate()
                        .map(|(idx, (begin, end))| {
                            let inp = Variable::new(format!("inp{}", idx));
                            let ge = Ge::make(&inp, begin);
                            let lt = Lt::make(&inp, input.dims[idx + 2].end() - end);
                            bitand(ge, lt)
                        })
                        .collect::<Vec<PrimeExpr>>();
                    let cond = all(&cmps);

                    let g = PrimeExpr::Variable(Variable::new(format!("ax1")));
                    let o = PrimeExpr::Variable(Variable::new(format!("ax2")));
                    let mut kernel_begins = vec![&g * &out_channels_per_group];
                    let mut kernel_steps = vec![(1i64).into()];
                    let mut axes = vec![o.clone()];
                    let mut strides = vec![
                        PrimeExpr::Load(Load::make(&format!("%{}.s", kernel.out_id), 0i64))
                    ];
                    for (idx, dim) in kernel_dims.iter().enumerate() {
                        kernel_begins.push((0i64).into());
                        kernel_steps.push((1i64).into());
                        axes.push(dim.var().to_prime_expr());
                        strides.push(
                            Load::make(&format!("%{}.s", kernel.out_id), (idx as i64) + 1).into()
                        );
                    }
                    let kernel_loaded = TensorLoad::make(
                        Variable::new(format!("%{}_val", kernel.out_id)),
                        kernel_begins,
                        axes,
                        kernel_steps,
                        strides,
                        vec![]
                    );
                    let if_stage = If {
                        cond,
                        true_bodys: vec![
                            Body::Stmt(
                                LetStmt::make(
                                    &Variable::new(format!("%{}_val", kernel.out_id)),
                                    kernel_loaded,
                                    false,
                                    Stmt::None
                                ).into()
                            )
                        ],
                        false_bodys: vec![],
                        id,
                        input: input.id,
                    };
                    lets_.push(Body::If(if_stage.clone()));
                    if is_output {
                        let sum = ReduceStage {
                            dims: kernel_dims,
                            bodys: lets_,
                            id: kernel.id,
                            inits: vec![
                                Body::Stmt(
                                    Stmt::AllocaStmt(
                                        AllocaStmt::make(
                                            &Variable::new(format!("%{}_val_ptr", id)),
                                            PrimitiveType::Dtype(kernel.dtype),
                                            1i64,
                                            Stmt::None
                                        )
                                    ).into()
                                ),
                                Body::Stmt(
                                    Stmt::StoreStmt(
                                        StoreStmt::make(
                                            &Variable::make(&format!("%{}_val_ptr", id)),
                                            0i64,
                                            dtype_zero(kernel.dtype)
                                        )
                                    ).into()
                                )
                            ],
                            posts: vec![add_bias],
                            input: kernel.id,
                        };
                        let stage = Stage {
                            dims,
                            bodys: vec![Body::ReduceStage(sum)],
                            dtype: kernel.dtype,
                            id,
                            out_id: id,
                        };
                        Body::Stage(stage)
                    } else {
                        let sum = ReduceStage {
                            dims: kernel_dims,
                            bodys: lets_,
                            id: kernel.id,
                            inits: vec![
                                Body::Stmt(
                                    Stmt::AllocaStmt(
                                        AllocaStmt::make(
                                            &Variable::new(format!("%{}_val_ptr", id)),
                                            PrimitiveType::Dtype(kernel.dtype),
                                            1i64,
                                            Stmt::None
                                        )
                                    ).into()
                                ),
                                Body::Stmt(
                                    Stmt::StoreStmt(
                                        StoreStmt::make(
                                            &Variable::make(&format!("%{}_val_ptr", id)),
                                            0i64,
                                            dtype_zero(kernel.dtype)
                                        )
                                    ).into()
                                )
                            ],
                            posts: vec![add_bias],
                            input: kernel.id,
                        };
                        let stage = Stage {
                            dims,
                            bodys: vec![Body::ReduceStage(sum)],
                            dtype: kernel.dtype,
                            id,
                            out_id: id,
                        };
                        Body::Stage(stage)
                    }
                }
                _ => panic!("The input should be a stage at {}", caller),
            }
        };
        let ret = Tensor {
            shape,
            inputs: Arc::new(vec![input.id, weight.id]),
            span: caller,
            id,
            dtype: weight.dtype.clone(),
            strides_cal: Arc::new(move |_: Vec<StridesCal>| {
                Arc::new(move |_: &HashMap<Arc<String>, i64>| { vec![] })
            }),
            body_gen: Arc::new(body),
        };
        self.nodes.borrow_mut().insert(id, ret.clone());
        ret
    }
}
