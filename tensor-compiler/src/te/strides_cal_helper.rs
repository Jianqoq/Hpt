use std::{ collections::HashMap, sync::Arc };

use tensor_common::{
    shape_utils::try_pad_shape,
    slice::{ slice_process, Slice },
    strides_utils::{ preprocess_strides, shape_to_strides },
};

use crate::{ halide::prime_expr::PrimeExpr, te::idx_evaluator::IdxEvaluator };

use super::hstrides::HStrides;

pub fn binary_strides_cal(
    lhs_shape: Arc<Vec<PrimeExpr>>,
    rhs_shape: Arc<Vec<PrimeExpr>>,
    res_shape: Arc<Vec<PrimeExpr>>,
    lhs_strides_cal: Arc<dyn Fn(&HashMap<Arc<String>, i64>) -> Vec<HStrides>>,
    rhs_strides_cal: Arc<dyn Fn(&HashMap<Arc<String>, i64>) -> Vec<HStrides>>
) -> Arc<dyn Fn(&HashMap<Arc<String>, i64>) -> Vec<HStrides>> {
    Arc::new(move |map: &HashMap<Arc<String>, i64>| {
        let lhs_strides = lhs_strides_cal(map);
        let rhs_strides = rhs_strides_cal(map);
        assert_eq!(lhs_strides.len(), rhs_strides.len());

        let lhs_real_shape = lhs_shape
            .iter()
            .map(|x| { IdxEvaluator::new(map).eval(x) })
            .collect::<Vec<i64>>();

        let rhs_real_shape = rhs_shape
            .iter()
            .map(|x| { IdxEvaluator::new(map).eval(x) })
            .collect::<Vec<i64>>();

        let res_real_shape = res_shape
            .iter()
            .map(|x| { IdxEvaluator::new(map).eval(x) })
            .collect::<Vec<i64>>();

        let mut lhs_strides_vec = vec![];
        for strides in lhs_strides.into_iter() {
            let masked = &strides[..strides.strides.len() - strides.reduced_dim];
            assert!(masked.len() == lhs_real_shape.len());
            let padded = try_pad_shape(&lhs_real_shape, res_real_shape.len());
            let new = preprocess_strides::<_, _, i64>(&padded, masked);
            let mut new_strides = vec![];
            for i in new.iter() {
                new_strides.push(*i);
            }
            for i in strides[strides.strides.len() - strides.reduced_dim..].iter() {
                new_strides.push(*i);
            }
            let new = HStrides {
                strides: new_strides,
                reduced_dim: strides.reduced_dim,
                offset: strides.offset,
            };
            lhs_strides_vec.push(new);
        }
        let mut rhs_strides_vec = vec![];
        for strides in rhs_strides.into_iter() {
            let masked = &strides[..strides.strides.len() - strides.reduced_dim];
            assert!(masked.len() == rhs_real_shape.len());
            let padded = try_pad_shape(&rhs_real_shape, res_real_shape.len());
            let new = preprocess_strides::<_, _, i64>(&padded, masked);
            let mut new_strides = vec![];
            for i in new.iter() {
                new_strides.push(*i);
            }
            for i in strides[strides.strides.len() - strides.reduced_dim..].iter() {
                new_strides.push(*i);
            }
            let new = HStrides {
                strides: new_strides,
                reduced_dim: strides.reduced_dim,
                offset: strides.offset,
            };
            rhs_strides_vec.push(new);
        }
        lhs_strides_vec.extend(rhs_strides_vec);
        lhs_strides_vec
    })
}

pub fn reduce_strides_cal(
    prev_func: Arc<dyn Fn(&HashMap<Arc<String>, i64>) -> Vec<HStrides>>,
    axes: Vec<usize>
) -> Arc<dyn Fn(&HashMap<Arc<String>, i64>) -> Vec<HStrides>> {
    Arc::new(move |map: &HashMap<Arc<String>, i64>| {
        let prev_strides = prev_func(map);
        let mut ret = vec![];
        for strides in prev_strides.into_iter() {
            let masked = &strides[..strides.strides.len() - strides.reduced_dim];
            let mut new_strides = vec![];
            for i in 0..masked.len() {
                if axes.contains(&i) {
                    continue;
                }
                new_strides.push(masked[i]);
            }
            for i in 0..masked.len() {
                if axes.contains(&i) {
                    new_strides.push(masked[i]);
                }
            }
            for i in strides[strides.strides.len() - strides.reduced_dim..].iter() {
                new_strides.push(*i);
            }
            assert!(new_strides.len() == strides.strides.len());
            let new = HStrides {
                strides: new_strides,
                reduced_dim: strides.reduced_dim + axes.len(),
                offset: strides.offset,
            };
            ret.push(new);
        }
        ret
    })
}

pub fn elementwise_strides_cal(
    prev_func: Arc<dyn Fn(&HashMap<Arc<String>, i64>) -> Vec<HStrides>>
) -> Arc<dyn Fn(&HashMap<Arc<String>, i64>) -> Vec<HStrides>> {
    Arc::new(move |map: &HashMap<Arc<String>, i64>| { prev_func(map) })
}

pub fn slice_strides_cal(
    shape: Arc<Vec<PrimeExpr>>,
    selections: Vec<(PrimeExpr, PrimeExpr, PrimeExpr)>,
    prev_func: Arc<dyn Fn(&HashMap<Arc<String>, i64>) -> Vec<HStrides>>
) -> Arc<dyn Fn(&HashMap<Arc<String>, i64>) -> Vec<HStrides>> {
    Arc::new(move |map: &HashMap<Arc<String>, i64>| {
        let selections = selections
            .iter()
            .map(|(start, end, step)| (
                IdxEvaluator::new(map).eval(start),
                IdxEvaluator::new(map).eval(end),
                IdxEvaluator::new(map).eval(step),
            ))
            .collect::<Vec<(i64, i64, i64)>>();
        let real_shape = shape
            .iter()
            .map(|x| { IdxEvaluator::new(map).eval(x) })
            .collect::<Vec<i64>>();
        let real_strides = shape_to_strides(&real_shape); // we always aloccate memory before we slice
        let (_, strides, offset) = slice_process(
            real_shape,
            real_strides.inner().clone(),
            &selections
                .into_iter()
                .map(|(x, y, z)| Slice::StepByRangeFromTo((x, y, z)))
                .collect::<Vec<Slice>>(),
            1
        ).expect("slice process failed");
        let mut strideses = prev_func(map);
        assert!(strideses.len() == 1);
        let hstrides = HStrides {
            strides,
            reduced_dim: 0,
            offset,
        };
        strideses.push(hstrides);
        strideses
    })
}
