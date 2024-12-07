use super::cfg::CFG;

pub(crate) struct _NodeEliminator<'a> {
    pub(crate) cfg: &'a mut CFG,
    pub(crate) current_assignment: Option<syn::Ident>,
}

impl<'ast> syn::visit::Visit<'ast> for _NodeEliminator<'ast> {}
