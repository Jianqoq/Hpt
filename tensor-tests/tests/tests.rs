#[test]
fn test_binary_expands() {
    macrotest::expand("tests/expands/binary.rs");
}

#[test]
fn test_expr_method_expands() {
    macrotest::expand("tests/expands/expr_method_call.rs");
}

#[test]
fn test_if_statements() {
    macrotest::expand("tests/control_flows/if_statements.rs");
}

#[test]
fn test_for_loop() {
    macrotest::expand("tests/control_flows/for_loop.rs");
}

#[test]
fn test_match() {
    macrotest::expand("tests/control_flows/match.rs");
}

#[test]
fn test_control_flow_mix() {
    macrotest::expand("tests/control_flows/mix.rs");
}

#[test]
fn test_while_loop() {
    macrotest::expand("tests/control_flows/while_loop.rs");
}

#[test]
fn test_let_lhs() {
    macrotest::expand("tests/let_expr/let_lhs.rs");
}

#[test]
fn test_let_rhs() {
    macrotest::expand("tests/let_expr/let_rhs.rs");
}

#[test]
fn test_macros() {
    macrotest::expand("tests/macros/*.rs");
}
