

#[test]
fn test_expands() {
    macrotest::expand("tests/expands/*.rs");
}
#[test]
fn test_control_flows() {
    macrotest::expand("tests/control_flows/*.rs");
}

#[test]
fn test_if_statements() {
    macrotest::expand("tests/control_flows/if_statements.rs");
}

#[test]
fn test_match_statements() {
    macrotest::expand("tests/control_flows/match.rs");
}

#[test]
fn test_let_lhs() {
    macrotest::expand("tests/let_lhs/*.rs");
}

#[test]
fn test_macros() {
    macrotest::expand("tests/macros/*.rs");
}
