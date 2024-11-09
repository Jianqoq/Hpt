use quote::ToTokens;

#[derive(Clone, Eq, PartialEq, Hash)]
pub(crate) struct Unary<'ast> {
    pub(crate) method: &'ast syn::Ident,
    pub(crate) operand: syn::Ident,
    pub(crate) output: syn::Ident,
}

impl<'ast> std::fmt::Debug for Unary<'ast> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} = {}({})",
            self.output.to_token_stream().to_string(),
            self.method.to_token_stream().to_string(),
            self.operand.to_token_stream().to_string()
        )
    }
}

#[derive(Clone, Eq, PartialEq, Hash)]
pub(crate) struct Binary {
    pub(crate) method: syn::Ident,
    pub(crate) left: syn::Ident,
    pub(crate) right: syn::Ident,
    pub(crate) output: syn::Ident,
}

impl std::fmt::Debug for Binary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} = {}({}, {})",
            {
                self.output.to_token_stream().to_string()
            },
            self.method.to_token_stream().to_string(),
            self.left.to_token_stream().to_string(),
            self.right.to_token_stream().to_string()
        )
    }
}

#[derive(Clone, Eq, PartialEq, Hash)]
pub(crate) enum Node<'ast> {
    Unary(Unary<'ast>, usize),
    Binary(Binary, usize),
}

impl<'ast> std::fmt::Debug for Node<'ast> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Node::Unary(unary, ..) => write!(f, "{:#?}", unary),
            Node::Binary(binary, ..) => write!(f, "{:#?}", binary),
        }
    }
}
