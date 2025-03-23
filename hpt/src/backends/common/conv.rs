pub(crate) fn cal_conv2d_output_shape(
    img_height: i64,
    img_width: i64,
    kh: i64,
    kw: i64,
    padding: &[(i64, i64); 2],
    stride: &[i64; 2],
    dilation: &[i64; 2],
) -> (i64, i64) {
    let out_height =
        (img_height + padding[0].0 + padding[0].1 - dilation[0] * (kh - 1) - 1) / stride[0] + 1;
    let out_width =
        (img_width + padding[1].0 + padding[1].1 - dilation[1] * (kw - 1) - 1) / stride[1] + 1;
    (out_height, out_width)
}
