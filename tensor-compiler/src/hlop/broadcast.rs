use tensor_types::dtype::Dtype;

use crate::{ halide::{ exprs::Int, prime_expr::PrimeExpr }, iter_var::IterVar };

pub fn predict_brocast_shape(
    lhs_shape: &[IterVar],
    rhs_shape: &[IterVar]
) -> (Vec<IterVar>, Vec<usize>, Vec<usize>) {
    let mut res_shape = vec![];
    let mut lhs_indices = vec![];
    let mut rhs_indices = vec![];
    let (lhs_start, rhs_start) = if lhs_shape.len() < rhs_shape.len() {
        (0..rhs_shape.len() - lhs_shape.len()).for_each(|x| {
            res_shape.push(rhs_shape[x].clone());
            lhs_indices.push(x);
        });
        (0, lhs_shape.len())
    } else if lhs_shape.len() > rhs_shape.len() {
        (0..lhs_shape.len() - rhs_shape.len()).for_each(|x| {
            res_shape.push(lhs_shape[x].clone());
            rhs_indices.push(x);
        });
        (rhs_shape.len(), 0)
    } else {
        (0, 0)
    };
    let one = PrimeExpr::Int(Int::make(Dtype::I64, 1));
    lhs_shape[lhs_start..]
        .iter()
        .zip(rhs_shape[rhs_start..].iter())
        .enumerate()
        .for_each(|(idx, (x, y))| {
            if x.end() == &one {
                res_shape.push(y.clone());
                rhs_indices.push(idx + rhs_start);
            } else if y.end() == &one {
                res_shape.push(x.clone());
                lhs_indices.push(idx + lhs_start);
            } else if x.end() - x.start() == y.end() - x.start() {
                res_shape.push(x.clone());
                lhs_indices.push(idx + lhs_start);
                rhs_indices.push(idx + rhs_start);
            } else {
                panic!("Incompatible shapes. {} and {}", x, y);
            }
        });
    (res_shape, lhs_indices, rhs_indices)
}
