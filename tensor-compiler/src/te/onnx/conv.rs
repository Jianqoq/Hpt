use std::{ collections::HashMap, panic::Location, sync::Arc };

use crate::{
    halide::{
        alloca_stmt::AllocaStmt,
        exprs::{ Ge, Load, Lt },
        let_stmt::LetStmt,
        passes::const_fold::ConstFold,
        prime_expr::PrimeExpr,
        primitive_type::PrimitiveType,
        stmt::Stmt,
        store_stmt::StoreStmt,
        tensor_load::TensorLoad,
        utils::{ all, bitand, dtype_zero, floor, store_with_dims, store_with_idx, var },
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
        let kernels_per_group = &weight.shape[0] / &groups;
        let in_channdels_per_group = weight.shape[1].clone();
        let body = move |inputs: Vec<Body>, is_output: bool, id: usize| {
            match (&inputs[0], &inputs[1]) {
                (Body::Stage(input), Body::Stage(kernel)) => {
                    let mut kernel_dims = Vec::with_capacity(kernel.dims[2..].len());
                    for (idx, i) in kernel.dims.iter().enumerate().skip(2) {
                        let mut i = i.clone();
                        i.set_var(var(format!("{}red{}", kernel.id, idx)));
                        kernel_dims.push(i);
                    }
                    let sum_ptr = var(format!("%{}_val_ptr", id));
                    let add_bias = if let Some(bias) = bias_id {
                        let load_sum: PrimeExpr = Load::make(&sum_ptr, 0i64).into();
                        let bias_val: PrimeExpr = var(format!("%{}_val", bias)).into();
                        let add = load_sum + bias_val;
                        Body::Stmt(store_with_idx(format!("%{}_val_ptr", id), 0i64, add))
                    } else {
                        Body::Stmt(Stmt::None)
                    };
                    let mut dims = outer_dims.as_ref().clone();
                    dims[1] = groups.clone();
                    dims.insert(2, kernels_per_group.clone());
                    dims.insert(3, in_channdels_per_group.clone());
                    let dims = dims
                        .into_iter()
                        .enumerate()
                        .map(|(idx, x)| { IterVar::new(0i64, x, 1i64, var(format!("ax{}", idx))) })
                        .collect::<Vec<IterVar>>();
                    let inp_begins = dims
                        .iter()
                        .skip(4)
                        .map(|x| x.var().to_prime_expr())
                        .collect::<Vec<_>>();
                    let kernel_begins = kernel_dims
                        .iter()
                        .map(|x| x.var().to_prime_expr())
                        .collect::<Vec<_>>();
                    let begins = inp_begins
                        .iter()
                        .zip(kernel_begins.iter())
                        .zip(steps.iter())
                        .zip(dilations.iter())
                        .zip(pads.iter())
                        .map(|((((x, y), z), w), pad)| {
                            x.clone() * z.to_prime_expr() +
                                y.clone() * w.to_prime_expr() -
                                pad.0.clone()
                        })
                        .collect::<Vec<_>>();
                    let mut lets_ = begins
                        .iter()
                        .enumerate()
                        .map(|(i, x)|
                            Body::Stmt(
                                Stmt::LetStmt(
                                    LetStmt::make(&var(format!("inp{}", i)), x, false, Stmt::None)
                                )
                            )
                        )
                        .collect::<Vec<Body>>();
                    lets_.push(
                        Body::Stmt(
                            Stmt::LetStmt(
                                LetStmt::make(
                                    &var(format!("%{}_val", id)),
                                    Load::make(&sum_ptr, 0i64),
                                    false,
                                    Stmt::None
                                )
                            )
                        )
                    );
                    let cmps = (0..pads.len())
                        .map(|idx| {
                            let inp = var(format!("inp{}", idx));
                            let ge = Ge::make(&inp, 0i64);
                            let lt = Lt::make(&inp, input.dims[idx + 2].end().clone());
                            bitand(ge, lt)
                        })
                        .collect::<Vec<PrimeExpr>>();

                    let g = PrimeExpr::Variable(var(format!("ax1")));
                    let o = PrimeExpr::Variable(var(format!("ax2")));
                    let mut kernel_begins = vec![o.clone(), (0i64).into()];
                    let mut kernel_steps = vec![kernels_per_group.clone(), (1i64).into()];
                    let mut axes = vec![g.clone(), dims[3].var().to_prime_expr()];
                    let mut strides = vec![
                        PrimeExpr::Load(Load::make(&format!("%{}.s", kernel.out_id), 0i64)),
                        PrimeExpr::Load(Load::make(&format!("%{}.s", kernel.out_id), 1i64))
                    ];
                    for idx in 0..kernel_dims.len() {
                        kernel_begins.push((0i64).into());
                        kernel_steps.push((1i64).into());
                        axes.push(var(format!("inp{}", idx)).into());
                        strides.push(
                            Load::make(&format!("%{}.s", kernel.out_id), (idx as i64) + 2).into()
                        );
                    }
                    let kernel_loaded = TensorLoad::make(
                        var(format!("%{}_val", kernel.out_id)),
                        kernel_begins,
                        axes,
                        kernel_steps,
                        strides,
                        vec![]
                    );

                    let mut inp_begins = vec![(0i64).into(), dims[3].var().to_prime_expr()];
                    let mut inp_steps = vec![(1i64).into(), in_channdels_per_group.clone()];
                    let mut axes = vec![dims[0].var().to_prime_expr(), g.clone()];
                    let mut strides = vec![
                        PrimeExpr::Load(Load::make(&format!("%{}.s", input.out_id), 0i64)),
                        PrimeExpr::Load(Load::make(&format!("%{}.s", input.out_id), 1i64))
                    ];

                    for idx in 0..input.dims.len() - 2 {
                        inp_begins.push((0i64).into());
                        inp_steps.push((1i64).into());
                        axes.push(var(format!("inp{}", idx)).into());
                        strides.push(
                            Load::make(&format!("%{}.s", input.out_id), (idx as i64) + 2).into()
                        );
                    }

                    let input_loaded = TensorLoad::make(
                        var(format!("%{}_val", input.out_id)),
                        inp_begins,
                        axes,
                        inp_steps,
                        strides,
                        vec![]
                    );

                    let mul =
                        var(format!("%{}_val", kernel.out_id)) *
                        var(format!("%{}_val", input.out_id));
                    let add = PrimeExpr::Variable(var(format!("%{}_val", id))) + mul;

                    let if_stage = If {
                        cond: all(&cmps),
                        true_bodys: vec![
                            Body::Stmt(
                                LetStmt::make(
                                    &var(format!("%{}_val", kernel.out_id)),
                                    kernel_loaded,
                                    false,
                                    Stmt::None
                                ).into()
                            ),
                            Body::Stmt(
                                LetStmt::make(
                                    &var(format!("%{}_val", input.out_id)),
                                    input_loaded,
                                    false,
                                    Stmt::None
                                ).into()
                            ),
                            Body::Stmt(store_with_idx(format!("%{}_val_ptr", id), 0i64, add))
                        ],
                        false_bodys: vec![],
                        id,
                        input: input.id,
                    };
                    lets_.push(Body::If(if_stage.clone()));
                    if is_output {
                        let mut store_begins = vec![(0i64).into(), o.clone()];
                        let mut store_steps = vec![(1i64).into(), kernels_per_group.clone()];
                        let mut store_axes = vec![
                            dims[0].var().to_prime_expr(),
                            dims[1].var().to_prime_expr()
                        ];
                        let mut store_strides = vec![
                            PrimeExpr::Load(Load::make(&format!("%{}.s", id), 0i64)),
                            PrimeExpr::Load(Load::make(&format!("%{}.s", id), 1i64))
                        ];
                        for idx in (0..dims.len()).skip(4) {
                            store_begins.push((0i64).into());
                            store_steps.push((1i64).into());
                            store_axes.push(var(format!("ax{}", idx)).into());
                            store_strides.push(
                                Load::make(&format!("%{}.s", id), (idx as i64) - 2).into()
                            );
                        }

                        let sum = ReduceStage {
                            dims: kernel_dims,
                            bodys: lets_,
                            id: kernel.id,
                            inits: vec![
                                Body::Stmt(
                                    Stmt::AllocaStmt(
                                        AllocaStmt::make(
                                            &var(format!("%{}_val_ptr", id)),
                                            PrimitiveType::Dtype(kernel.dtype),
                                            1i64,
                                            Stmt::None
                                        )
                                    ).into()
                                ),
                                Body::Stmt(
                                    store_with_idx(
                                        format!("%{}_val_ptr", id),
                                        0i64,
                                        dtype_zero(kernel.dtype)
                                    )
                                )
                            ],
                            posts: vec![add_bias],
                            input: kernel.id,
                        };
                        let stage = Stage {
                            dims,
                            bodys: vec![
                                Body::ReduceStage(sum),
                                Body::Stmt(
                                    Stmt::StoreStmt(
                                        StoreStmt::make(
                                            format!("%{}", id),
                                            store_begins,
                                            store_axes,
                                            store_steps,
                                            store_strides,
                                            Load::make(&format!("%{}_val_ptr", id), 0i64)
                                        )
                                    )
                                )
                            ],
                            dtype: kernel.dtype,
                            id,
                            out_id: id,
                        };
                        Body::Stage(stage)
                    } else {
                        let let_stmt = Body::Stmt(
                            LetStmt::make(
                                &var(format!("%{}_val", id)),
                                Load::make(&sum_ptr, 0i64),
                                false,
                                Stmt::None
                            ).into()
                        );
                        let sum = ReduceStage {
                            dims: kernel_dims,
                            bodys: lets_,
                            id: kernel.id,
                            inits: vec![
                                Body::Stmt(
                                    Stmt::AllocaStmt(
                                        AllocaStmt::make(
                                            &var(format!("%{}_val_ptr", id)),
                                            PrimitiveType::Dtype(kernel.dtype),
                                            1i64,
                                            Stmt::None
                                        )
                                    ).into()
                                ),
                                Body::Stmt(
                                    store_with_idx(
                                        format!("%{}_val_ptr", id),
                                        0i64,
                                        dtype_zero(kernel.dtype)
                                    )
                                )
                            ],
                            posts: vec![add_bias],
                            input: kernel.id,
                        };
                        let stage = Stage {
                            dims,
                            bodys: vec![Body::ReduceStage(sum), let_stmt],
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
