#[derive(Debug)]
pub(crate) struct Unary<'ast> {
    pub(crate) method: &'ast syn::Ident,
    pub(crate) operand: &'ast syn::Expr,
    pub(crate) outputs: Vec<syn::Ident>,
}

#[derive(Debug)]
pub(crate) struct Binary {
    pub(crate) method: syn::Ident,
    pub(crate) left: syn::Ident,
    pub(crate) right: syn::Ident,
    pub(crate) outputs: Vec<syn::Ident>,
}

#[derive(Debug)]
pub(crate) enum Node<'ast> {
    Unary(Unary<'ast>),
    Binary(Binary),
}
