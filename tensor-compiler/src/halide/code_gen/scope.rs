use std::sync::Arc;

use std::collections::{ HashMap, HashSet };
use tensor_llvm::types::values::BasicValue;

use crate::halide::primitive_type::PrimitiveType;

pub struct Scope {
    variables: HashMap<Arc<String>, BasicValue>,
    types: HashMap<BasicValue, PrimitiveType>,
    mutables: HashSet<Arc<String>>,
}

impl Scope {
    pub fn variables(&self) -> &HashMap<Arc<String>, BasicValue> {
        &self.variables
    }
    pub fn types(&self) -> &HashMap<BasicValue, PrimitiveType> {
        &self.types
    }
    pub fn mutables(&self) -> &HashSet<Arc<String>> {
        &self.mutables
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
                mutables: HashSet::new(),
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
    pub fn is_mutable(&self, name: &String) -> bool {
        for scope in self.scopes.iter().rev() {
            if scope.mutables.contains(name) {
                return true;
            }
        }
        false
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
    pub fn insert_mutable(&mut self, name: &Arc<String>) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.mutables.insert(name.clone());
        }
    }
    pub fn push_scope(&mut self) {
        self.scopes.push(Scope {
            variables: HashMap::new(),
            types: HashMap::new(),
            mutables: HashSet::new(),
        });
    }
    pub fn pop_scope(&mut self) {
        self.scopes.pop();
    }
    pub fn inner(&self) -> &Vec<Scope> {
        &self.scopes
    }
}
