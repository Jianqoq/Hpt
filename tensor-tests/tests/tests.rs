

#[test]
fn test_expands() {
    macrotest::expand("tests/expands/*.rs");
}
#[test]
fn test_control_flows() {
    macrotest::expand("tests/control_flows/*.rs");
}