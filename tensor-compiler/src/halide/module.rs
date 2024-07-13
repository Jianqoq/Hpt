use std::{ fmt::Display, sync::Arc };

use hashbrown::HashSet;

use crate::halide::printer::_IRPrinter;

use super::{ primitive_type::PrimitiveType, stmt::Stmt };

#[derive(Eq, PartialEq)]
pub struct Function {
    pub(crate) ty: FunctionType,
    pub(crate) name: Arc<String>,
    pub(crate) body: Stmt,
}

impl std::hash::Hash for Function {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ty.hash(state);
        self.name.hash(state);
    }
}

impl Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "fn {}(", self.name)?;
        for (i, (name, r#type)) in self.ty.args.iter().enumerate() {
            if i != 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}: {}", name, r#type)?;
        }
        write!(f, ") -> {}", self.ty.ret_ty)?;
        write!(f, " {{\n")?;
        write!(f, "{}", _IRPrinter::new(1).print_stmt_str(&self.body))?;
        write!(f, "}}")
    }
}

#[derive(Clone, Hash, Eq, PartialEq)]
pub struct FunctionType {
    ret_ty: PrimitiveType,
    args: Arc<Vec<(String, PrimitiveType)>>,
}

impl FunctionType {
    pub fn new<T: IntoIterator<Item: Into<(String, PrimitiveType)>>>(
        ret_ty: PrimitiveType,
        args: T
    ) -> Self {
        FunctionType {
            ret_ty,
            args: args
                .into_iter()
                .map(|x| x.into())
                .collect::<Vec<_>>()
                .into(),
        }
    }
}

pub struct Module {
    name: Arc<String>,
    fns: HashSet<Function>,
}

impl Module {
    pub fn new(name: String) -> Self {
        Module {
            name: name.into(),
            fns: HashSet::new(),
        }
    }

    pub fn add_function(&mut self, ty: FunctionType, name: &str) {
        self.fns.insert(Function { ty, body: Stmt::None, name: name.to_string().into() });
    }
}
