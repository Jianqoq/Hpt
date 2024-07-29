use std::{ collections::HashMap, fmt::Display, sync::Arc };

use crate::halide::code_gen::code_gen::CodeGen;

use super::executable::Executable;

pub enum StringKey<'a> {
    Str(String),
    ArcStr(Arc<String>),
    Ref(&'a str),
}

impl Display for StringKey<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StringKey::Str(s) => write!(f, "{}", s),
            StringKey::ArcStr(s) => write!(f, "{}", s),
            StringKey::Ref(s) => write!(f, "{}", s),
        }
    }
}

impl<'a> Into<StringKey<'a>> for String {
    fn into(self) -> StringKey<'a> {
        StringKey::Str(self)
    }
}

impl<'a> Into<StringKey<'a>> for Arc<String> {
    fn into(self) -> StringKey<'a> {
        StringKey::ArcStr(self)
    }
}

impl<'a> Into<StringKey<'a>> for &'a str {
    fn into(self) -> StringKey<'a> {
        StringKey::Ref(self)
    }
}

pub struct EmptyExecutable {
    codegen: CodeGen,
}

impl EmptyExecutable {
    pub fn new(codegen: CodeGen) -> Self {
        Self { codegen }
    }

    pub fn into_executable<'a>(
        self,
        vars_map: HashMap<impl Into<StringKey<'a>>, i64>
    ) -> Executable {
        let map = vars_map
            .into_iter()
            .map(|(k, v)| {
                let k: StringKey<'a> = k.into();
                (k.to_string().into(), v)
            })
            .collect::<HashMap<Arc<String>, i64>>();
        self.codegen.into_executable(map)
    }
}
