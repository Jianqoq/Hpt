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

impl Error {
    pub(crate) fn to_anyhow_error(&self) -> anyhow::Error {
        let syn_err = syn::Error::new(self.span(), self.to_string());
        syn_err.into()
    }
    pub(crate) fn span(&self) -> proc_macro2::Span {
        match self {
            Self::ExpectedIdentifier(span, _) => *span,
            Self::ExpectedPath(span, _) => *span,
            Self::ExpectedAssignment(span, _) => *span,
        }
    }
}
