use std::{ fmt::Display, sync::Arc };

use tensor_common::layout::Layout;
use tensor_types::dtype::Dtype;

use super::{_value::_Value, node::Expr};

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Value {
    dtype: Dtype,
    value: _Value,
}

impl Value {
    pub fn make<T: Into<_Value>>(dtype: Dtype, value: T) -> Self {
        Self {
            dtype,
            value: value.into(),
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Str {
    value: Arc<String>,
}

impl Display for Str {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct Variable {
    value: Arc<String>,
}

impl Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Cast {
    expr: Arc<Expr>,
    dtype: Dtype,
}

impl Display for Cast {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} as {})", self.expr, self.dtype)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Add {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for Add {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} + {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Sub {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for Sub {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} - {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Mul {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for Mul {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} * {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Div {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for Div {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} / {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Mod {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for Mod {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} % {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Min {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for Min {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "min({}, {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Max {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for Max {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "max({}, {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Eq {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for Eq {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} == {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Ne {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for Ne {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} != {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Lt {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for Lt {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} < {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Le {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for Le {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} <= {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Gt {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for Gt {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} > {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Ge {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for Ge {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} >= {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct And {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for And {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} && {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Or {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for Or {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} || {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Xor {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for Xor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} ^ {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Not {
    expr: Arc<Expr>,
}

impl Display for Not {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "!{}", self.expr)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Call {
    name: Arc<String>,
    args: Vec<Arc<Expr>>,
}

impl Display for Call {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}({})",
            self.name,
            self.args
                .iter()
                .map(|x| format!("{}", x))
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Select {
    cond: Arc<Expr>,
    true_value: Arc<Expr>,
    false_value: Arc<Expr>,
}

impl Display for Select {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} ? {} : {})", self.cond, self.true_value, self.false_value)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Let {
    var: Arc<Variable>,
    value: Arc<Expr>,
}

impl Display for Let {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "let {} = {}", self.var, self.value)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Tensor {
    name: Arc<String>,
    layout: Arc<Layout>,
    dtype: Dtype,
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}{{{:?},{}}}", self.name, self.layout.shape().inner(), self.dtype)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Alloc {
    shape: Arc<Expr>,
    dtype: Dtype,
}

impl Display for Alloc {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "alloc({}, {})", self.shape, self.dtype)
    }
}

macro_rules! impl_into_expr {
    ($struct:ident) => {
        impl Into<Expr> for $struct {
            fn into(self) -> Expr {
                Expr::$struct(self)
            }
        }
        
        impl Into<Expr> for &$struct {
            fn into(self) -> Expr {
                Expr::$struct(self.clone())
            }
        }
    };
}

impl_into_expr!(Value);
impl_into_expr!(Str);
impl_into_expr!(Variable);
impl_into_expr!(Cast);
impl_into_expr!(Add);
impl_into_expr!(Sub);
impl_into_expr!(Mul);
impl_into_expr!(Div);
impl_into_expr!(Mod);
impl_into_expr!(Min);
impl_into_expr!(Max);
impl_into_expr!(Eq);
impl_into_expr!(Ne);
impl_into_expr!(Lt);
impl_into_expr!(Le);
impl_into_expr!(Gt);
impl_into_expr!(Ge);
impl_into_expr!(And);
impl_into_expr!(Or);
impl_into_expr!(Xor);
impl_into_expr!(Not);
impl_into_expr!(Call);
impl_into_expr!(Select);
impl_into_expr!(Let);
impl_into_expr!(Tensor);
impl_into_expr!(Alloc);
