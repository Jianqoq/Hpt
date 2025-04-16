pub(crate) fn assert_f64(a: f64, b: f64, diff: f64) -> anyhow::Result<()> {
    let rel_diff = ((a - b) / (a.abs() + b.abs() + f64::EPSILON)).abs();
    if rel_diff > diff {
        return Err(anyhow::anyhow!(
            "{} != {} (relative_diff: {}).\n",
            a,
            b,
            rel_diff
        ));
    }
    Ok(())
}

#[allow(unused)]
#[must_use]
pub(crate) fn assert_f32(a: f32, b: f32, diff: f32) -> anyhow::Result<()> {
    let rel_diff = ((a - b) / (a.abs() + b.abs() + f32::EPSILON)).abs();
    if rel_diff > diff {
        return Err(anyhow::anyhow!(
            "{} != {} (relative_diff: {}).\n",
            a,
            b,
            rel_diff
        ));
    }
    Ok(())
}
