use hpt_core::Tensor;

pub(crate) fn assert_f64(
    a: f64,
    b: f64,
    diff: f64,
    res: &Tensor<f64>,
    tch_res: &tch::Tensor,
) -> anyhow::Result<()> {
    let rel_diff = ((a - b) / (a.abs() + b.abs() + f64::EPSILON)).abs();
    if rel_diff > diff {
        return Err(anyhow::anyhow!(
            "{} != {} (relative_diff: {}).\n res: {}\n res2: {}",
            a,
            b,
            rel_diff,
            res,
            tch_res
        ));
    }
    Ok(())
}

#[allow(unused)]
#[must_use]
pub(crate) fn assert_f32(
    a: f32,
    b: f32,
    diff: f32,
    res: &Tensor<f32>,
    tch_res: &tch::Tensor,
) -> anyhow::Result<()> {
    let rel_diff = ((a - b) / (a.abs() + b.abs() + f32::EPSILON)).abs();
    if rel_diff > diff {
        return Err(anyhow::anyhow!(
            "{} != {} (relative_diff: {}).\n res: {}\n res2: {}",
            a,
            b,
            rel_diff,
            res,
            tch_res
        ));
    }
    Ok(())
}
