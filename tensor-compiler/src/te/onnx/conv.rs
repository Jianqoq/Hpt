use std::{ collections::HashMap, panic::Location, sync::Arc };

use crate::{
    halide::{
        alloca_stmt::AllocaStmt, exprs::Load, passes::const_fold::ConstFold, prime_expr::PrimeExpr, primitive_type::PrimitiveType, stmt::Stmt, store_stmt::StoreStmt, utils::{ dtype_zero, floor }, variable::Variable
    },
    iter_var::IterVar,
    te::{ context::Context, stages::{ Body, ReduceStage, Stage }, tensor::{ StridesCal, Tensor } },
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
        let body = move |inputs: Vec<Body>, is_output: bool, id: usize| {
            match (&inputs[0], &inputs[1]) {
                (Body::Stage(_), Body::Stage(kernel)) => {
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
                    dims.insert(2, kernel.dims[1].end().clone());
                    let dims = dims
                        .into_iter()
                        .enumerate()
                        .map(|(idx, x)| {
                            IterVar::new(0i64, x, 1i64, Variable::new(format!("ax{}", idx)))
                        })
                        .collect();
                    if is_output {
                        let sum = ReduceStage {
                            dims: kernel_dims,
                            bodys: vec![],
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
                            bodys: vec![],
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
