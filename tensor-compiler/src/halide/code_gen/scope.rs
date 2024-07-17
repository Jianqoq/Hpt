use std::sync::Arc;

use std::collections::HashMap;
use tensor_llvm::types::values::BasicValue;

use crate::halide::primitive_type::PrimitiveType;

pub struct Scope {
    variables: HashMap<Arc<String>, BasicValue>,
    types: HashMap<BasicValue, PrimitiveType>,
}

impl Scope {
    pub fn variables(&self) -> &HashMap<Arc<String>, BasicValue> {
        &self.variables
    }
    pub fn types(&self) -> &HashMap<BasicValue, PrimitiveType> {
        &self.types
    }
}

pub struct ScopeStack {
    scopes: Vec<Scope>,
}

impl ScopeStack {
    pub fn new() -> Self {
        ScopeStack {
            scopes: vec![Scope {
                variables: HashMap::new(),
                types: HashMap::new(),
            }],
        }
    }
    pub fn find_type(&self, val: &BasicValue) -> Option<PrimitiveType> {
        for scope in self.scopes.iter().rev() {
            if let Some(dtype) = scope.types.get(val) {
                return Some(dtype.clone());
            }
        }
        None
    }
    pub fn find_variable(&self, name: &String) -> Option<BasicValue> {
        for scope in self.scopes.iter().rev() {
            if let Some(val) = scope.variables.get(name) {
                return Some(val.clone());
            }
        }
        None
    }
    pub fn declare_variable(&mut self, name: &Arc<String>, val: BasicValue, dtype: PrimitiveType) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.variables.insert(name.clone(), val);
            scope.types.insert(val, dtype);
        }
    }
    pub fn insert_type(&mut self, val: BasicValue, dtype: PrimitiveType) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.types.insert(val, dtype);
        }
    }
    pub fn insert_variable(&mut self, name: &Arc<String>, val: BasicValue) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.variables.insert(name.clone(), val);
        }
    }
    pub fn push_scope(&mut self) {
        self.scopes.push(Scope {
            variables: HashMap::new(),
            types: HashMap::new(),
        });
    }
    pub fn pop_scope(&mut self) {
        self.scopes.pop();
    }
    pub fn inner(&self) -> &Vec<Scope> {
        &self.scopes
    }
}
