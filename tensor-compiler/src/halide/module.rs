use std::{ fmt::Display, sync::Arc };

use hashbrown::{ HashMap, HashSet };

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
    pub(crate) ret_ty: PrimitiveType,
    pub(crate) args: Arc<Vec<(String, PrimitiveType)>>,
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
    pub(crate) name: Arc<String>,
    pub(crate) imports: HashSet<Arc<String>>,
    pub(crate) fns: HashMap<String, Function>,
}

impl Module {
    pub fn new<T: Into<String>>(name: T) -> Self {
        Module {
            name: name.into().into(),
            fns: HashMap::new(),
            imports: HashSet::new(),
        }
    }

    pub fn add_function(&mut self, ty: FunctionType, name: &str) {
        self.fns.insert(name.to_string(), Function {
            ty,
            body: Stmt::None,
            name: name.to_string().into(),
        });
    }
    pub fn get_function(&self, name: &str) -> Option<&Function> {
        self.fns.get(name)
    }
    pub fn get_function_mut(&mut self, name: &str) -> Option<&mut Function> {
        self.fns.get_mut(name)
    }
}

impl Display for Module {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "module {} {{\n", self.name)?;
        for import in self.imports.iter() {
            write!(f, "import \"{}\"\n", import)?;
        }
        for func in self.fns.values() {
            write!(f, "{}\n", func)?;
        }
        write!(f, "}}")
    }
}
