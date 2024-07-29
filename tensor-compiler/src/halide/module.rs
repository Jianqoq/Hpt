use std::{ fmt::Display, sync::Arc };

use std::collections::{ HashMap, HashSet };
use tensor_llvm::{
    context::context::Context,
    types::{ general_types::GeneralType, values::StructValue },
};
use tensor_types::dtype::Dtype;

use crate::halide::printer::_IRPrinter;
use crate::te::hstrides::HStrides;
use crate::te::schedule::Schedule;

use super::prime_expr::PrimeExpr;
use super::{ primitive_type::PrimitiveType, stmt::Stmt };

#[derive(Eq, PartialEq, Clone)]
pub struct Function {
    pub ty: FunctionType,
    pub name: Arc<String>,
    pub body: Stmt,
}

impl Function {
    pub fn ty(&self) -> &FunctionType {
        &self.ty
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn body(&self) -> &Stmt {
        &self.body
    }
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
    pub fn to_llvm_func_type(
        &self,
        ctx: &Context,
        tensor_type: StructValue
    ) -> tensor_llvm::types::types::FunctionType {
        let mut arg_types = Vec::new();
        for (_, ty) in self.args.iter() {
            arg_types.push(ty.to_llvm_type(ctx, tensor_type));
        }
        let ret_type = self.ret_ty.to_llvm_type(ctx, tensor_type);
        match ret_type {
            GeneralType::Bool(val) => val.fn_type(&arg_types, false),
            GeneralType::BoolPtr(val) => val.fn_type(&arg_types, false),
            GeneralType::I8(val) => val.fn_type(&arg_types, false),
            GeneralType::I8Ptr(val) => val.fn_type(&arg_types, false),
            GeneralType::U8(val) => val.fn_type(&arg_types, false),
            GeneralType::U8Ptr(val) => val.fn_type(&arg_types, false),
            GeneralType::I16(val) => val.fn_type(&arg_types, false),
            GeneralType::I16Ptr(val) => val.fn_type(&arg_types, false),
            GeneralType::U16(val) => val.fn_type(&arg_types, false),
            GeneralType::U16Ptr(val) => val.fn_type(&arg_types, false),
            GeneralType::I32(val) => val.fn_type(&arg_types, false),
            GeneralType::I32Ptr(val) => val.fn_type(&arg_types, false),
            GeneralType::U32(val) => val.fn_type(&arg_types, false),
            GeneralType::U32Ptr(val) => val.fn_type(&arg_types, false),
            GeneralType::I64(val) => val.fn_type(&arg_types, false),
            GeneralType::I64Ptr(val) => val.fn_type(&arg_types, false),
            GeneralType::U64(val) => val.fn_type(&arg_types, false),
            GeneralType::U64Ptr(val) => val.fn_type(&arg_types, false),
            GeneralType::BF16(val) => val.fn_type(&arg_types, false),
            GeneralType::BF16Ptr(val) => val.fn_type(&arg_types, false),
            GeneralType::F16(val) => val.fn_type(&arg_types, false),
            GeneralType::F16Ptr(val) => val.fn_type(&arg_types, false),
            GeneralType::F32(val) => val.fn_type(&arg_types, false),
            GeneralType::F32Ptr(val) => val.fn_type(&arg_types, false),
            GeneralType::F64(val) => val.fn_type(&arg_types, false),
            GeneralType::F64Ptr(val) => val.fn_type(&arg_types, false),
            GeneralType::Void(val) => val.fn_type(&arg_types, false),
            GeneralType::VoidPtr(val) => val.fn_type(&arg_types, false),
            GeneralType::Isize(val) => val.fn_type(&arg_types, false),
            GeneralType::IsizePtr(val) => val.fn_type(&arg_types, false),
            GeneralType::FunctionPtr(val) => val.fn_type(&arg_types, false),
            GeneralType::Array(val) => val.fn_type(&arg_types, false),
            GeneralType::ArrayPtr(val) => val.fn_type(&arg_types, false),
            GeneralType::Str(val) => val.fn_type(&arg_types, false),
            GeneralType::StrPtr(val) => val.fn_type(&arg_types, false),
            GeneralType::Struct(val) => val.fn_type(&arg_types, false),
            GeneralType::StructPtr(val) => val.fn_type(&arg_types, false),
            _ => unimplemented!(),
        }
    }
}

#[derive(Clone)]
pub struct FunctionMeta {
    pub(crate) function: Function,
    pub(crate) inputs_order: Vec<usize>,
    pub(crate) inputs_dtype: Vec<Dtype>,
    pub(crate) outputs_order: Vec<usize>,
    pub(crate) outputs_dtype: Vec<Dtype>,
    pub(crate) outs_shape: Vec<Arc<Vec<PrimeExpr>>>,
    pub(crate) inps_shape: Vec<Arc<Vec<PrimeExpr>>>,
    pub(crate) strides_cal: Arc<dyn Fn(&HashMap<Arc<String>, i64>) -> Vec<HStrides>>,
    pub(crate) vars_order: Vec<Arc<String>>,
}

#[derive(Clone)]
pub struct Module {
    pub(crate) name: Arc<String>,
    pub(crate) imports: HashSet<Arc<String>>,
    pub(crate) fns: HashMap<Arc<String>, FunctionMeta>,
    pub(crate) order: Vec<Arc<String>>,
}

impl Module {
    pub fn new<T: Into<String>>(name: T) -> Self {
        Module {
            name: name.into().into(),
            fns: HashMap::new(),
            imports: HashSet::new(),
            order: Vec::new(),
        }
    }
    pub fn set_order(&mut self, order: Vec<Arc<String>>) {
        self.order = order;
    }
    pub fn add_function2(&mut self, schedule: &Schedule) {
        let strides_cal = schedule.strides_cal.clone();
        let function = schedule.to_function();
        let inputs = schedule.inputs_order();
        let inputs_dtype = schedule.inputs_dtype();
        let outputs = schedule.outputs_order();
        let outputs_dtype = schedule.outputs_dtype();
        let outs_shape = schedule.outs_shape();
        let inps_shape = schedule.inps_shape();
        let vars_order = schedule.vars_order();
        self.fns.insert(function.name.clone(), FunctionMeta {
            function,
            inputs_order: inputs,
            inputs_dtype,
            outputs_order: outputs,
            outputs_dtype,
            outs_shape,
            inps_shape,
            strides_cal,
            vars_order,
        });
    }
    pub fn get_function(&self, name: &Arc<String>) -> Option<&Function> {
        self.fns.get(name).map(|x| &x.function)
    }
    pub fn get_function_mut(&mut self, name: &Arc<String>) -> Option<&mut Function> {
        self.fns.get_mut(name).map(|x| &mut x.function)
    }
}

impl Display for Module {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "module {} {{\n", self.name)?;
        for import in self.imports.iter() {
            write!(f, "import \"{}\"\n", import)?;
        }
        for func in self.fns.values() {
            write!(f, "{}\n", func.function)?;
        }
        write!(f, "}}")
    }
}
