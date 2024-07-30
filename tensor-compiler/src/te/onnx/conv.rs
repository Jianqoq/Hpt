use std::panic::Location;

use crate::{
    halide::prime_expr::PrimeExpr,
    te::{ context::Context, tensor::Tensor },
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
        dilations: Option<i64>,
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
            steps.iter()
                .map(|x| x.to_prime_expr())
                .collect()
        } else {
            vec![1i64.into(); tmp.len()]
        };
        let dilations = dilations.unwrap_or(1);
        let group = group.unwrap_or(1);
        todo!()
    }
}
