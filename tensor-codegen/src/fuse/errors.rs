use thiserror::Error;

#[derive(Error, Debug)]
pub(crate) enum Error {
    #[error("Internal {1} error: Expected an identifier")] ExpectedIdentifier(
        proc_macro2::Span,
        &'static str,
    ),
    #[error("Internal {1} error: Expected a path")] ExpectedPath(proc_macro2::Span, &'static str),
    #[error("Internal {1} error: Expected has an assignment variable")] ExpectedAssignment(
        proc_macro2::Span,
        &'static str,
    ),
}
