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
    #[error("Internal {1} error: {2} is not supported yet")] Unsupported(
        proc_macro2::Span,
        &'static str,
        String,
    ),
    #[error("Internal {1} error: original variable {2} not found")] OriginalVariableNotFound(
        proc_macro2::Span,
        &'static str,
        String,
    ),
    #[error("Internal {1} error: {2} didn't accumulate any expression")] ExprAccumulateError(
        proc_macro2::Span,
        &'static str,
        String,
    ),
    #[error("Internal {1} error: syn parsing `{2}` failed")] SynParseError(
        proc_macro2::Span,
        &'static str,
        String,
    ),
}

impl Error {
    pub(crate) fn to_anyhow_error(&self) -> anyhow::Error {
        let syn_err = syn::Error::new(self.span(), self.to_string());
        syn_err.into()
    }
    pub(crate) fn to_syn_error(&self) -> syn::Error {
        syn::Error::new(self.span(), self.to_string())
    }
    pub(crate) fn span(&self) -> proc_macro2::Span {
        match self {
            Self::ExpectedIdentifier(span, _) => *span,
            Self::ExpectedPath(span, _) => *span,
            Self::ExpectedAssignment(span, _) => *span,
            Self::Unsupported(span, _, _) => *span,
            Self::OriginalVariableNotFound(span, _, _) => *span,
            Self::ExprAccumulateError(span, _, _) => *span,
            Self::SynParseError(span, _, _) => *span,
        }
    }
}
