use quote::ToTokens;
use syn::spanned::Spanned;

use super::kernel_type::KernelType;

#[derive(Clone, Eq, PartialEq, Hash)]
pub(crate) enum Operand {
    Constant(syn::Expr),
    Variable(syn::Ident),
}

impl Operand {
    pub(crate) fn is_variable(&self) -> bool {
        matches!(self, Operand::Variable(_))
    }
    pub(crate) fn span(&self) -> proc_macro2::Span {
        match self {
            Operand::Constant(lit) => lit.span(),
            Operand::Variable(ident) => ident.span(),
        }
    }
}

impl std::fmt::Display for Operand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operand::Constant(lit) => write!(f, "{}", lit.to_token_stream().to_string()),
            Operand::Variable(ident) => write!(f, "{}", ident.to_token_stream().to_string()),
        }
    }
}

impl<'ast> ToTokens for Operand {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            Operand::Constant(lit) => lit.to_tokens(tokens),
            Operand::Variable(ident) => ident.to_tokens(tokens),
        }
    }
}

#[derive(Clone, Eq, PartialEq, Hash)]
pub(crate) struct Unary {
    pub(crate) method: syn::Ident,
    pub(crate) operand: Operand,
    pub(crate) args: Vec<Operand>,
    pub(crate) output: Operand,
    pub(crate) kernel_type: KernelType,
}

impl std::fmt::Debug for Unary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} = {}({}, {})",
            self.output.to_token_stream().to_string(),
            self.method.to_token_stream().to_string(),
            self.operand,
            self.args
                .iter()
                .map(|arg| arg.to_token_stream().to_string())
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

impl ToTokens for Unary {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let method = proc_macro2::Ident::new(&format!("_{}", self.method), self.method.span());
        let operand = &self.operand;
        let output = &self.output;
        if self.args.is_empty() {
            tokens.extend(
                quote::quote!(
                let #output = #operand.#method();
            )
            );
        } else {
            let args = self.args.iter();
            tokens.extend(
                quote::quote!(
                let #output = #operand.#method(#(#args),*);
            )
            );
        }
    }
}

#[derive(Clone, Eq, PartialEq, Hash)]
pub(crate) struct Binary {
    pub(crate) method: syn::Ident,
    pub(crate) left: Operand,
    pub(crate) right: Operand,
    pub(crate) output: Operand,
    pub(crate) kernel_type: KernelType,
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
            self.left,
            self.right
        )
    }
}

impl<'ast> ToTokens for Binary {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let method = proc_macro2::Ident::new(&format!("_{}", self.method), self.method.span());
        let left = &self.left;
        let right = &self.right;
        let output = &self.output;
        tokens.extend(quote::quote!(
             let #output = #left.#method(#right);
        ));
    }
}

#[derive(Clone, Eq, PartialEq, Hash)]
pub(crate) enum Node {
    Unary(Unary),
    Binary(Binary),
    Input(Operand),
}

impl std::fmt::Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Node::Unary(unary, ..) => write!(f, "{:#?}", unary),
            Node::Binary(binary, ..) => write!(f, "{:#?}", binary),
            Node::Input(input) => write!(f, "input({})", input.to_string()),
        }
    }
}
