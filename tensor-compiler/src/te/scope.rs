use std::sync::Arc;

use std::collections::HashSet;

use crate::halide::prime_expr::PrimeExpr;
use crate::halide::variable::Variable;

pub struct Scope {
    variables: HashSet<Variable>,
}

impl Scope {
    pub fn variables(&self) -> &HashSet<Variable> {
        &self.variables
    }
}

pub struct ScopeStack {
    scopes: Vec<Scope>,
}

impl ScopeStack {
    pub fn new() -> Self {
        ScopeStack {
            scopes: vec![Scope {
                variables: HashSet::new(),
            }],
        }
    }
    pub fn find_variable(&self, name: &Variable) -> Option<Variable> {
        for scope in self.scopes.iter().rev() {
            if let Some(val) = scope.variables.get(name) {
                return Some(val.clone());
            }
        }
        None
    }
    pub fn declare_variable(&mut self, name: &Variable) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.variables.insert(name.clone());
        }
    }
    pub fn insert_variable(&mut self, name: &Variable) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.variables.insert(name.clone());
        }
    }
    pub fn push_scope(&mut self) {
        self.scopes.push(Scope {
            variables: HashSet::new(),
        });
    }
    pub fn pop_scope(&mut self) {
        self.scopes.pop();
    }
    pub fn inner(&self) -> &Vec<Scope> {
        &self.scopes
    }
}
