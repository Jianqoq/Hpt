

#[test]
fn test_add() {
    use tensor_macros::infer_cal_res_type;
    infer_cal_res_type!(bool, i8, add);
}