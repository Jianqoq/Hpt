use crate::halide::traits::Accepter;

use super::{ stmt::Stmt, traits::{ AccepterMut, IRMutVisitor, IRMutateVisitor, IRVisitor } };

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Seq {
    stmts: Vec<Stmt>,
}

impl Seq {
    pub fn stmts(&self) -> &Vec<Stmt> {
        &self.stmts
    }

    pub fn make<T: Into<Vec<Stmt>>>(stmts: T) -> Self {
        Seq {
            stmts: stmts.into(),
        }
    }

    pub fn accept<V: IRVisitor>(&self, visitor: &V) {
        for stmt in &self.stmts {
            stmt.accept(visitor);
        }
    }

    pub fn accept_mutate<V: IRMutateVisitor>(&self, visitor: &mut V) {
        visitor.visit_seq_stmt(self);
    }

    pub fn accept_mut<V: IRMutVisitor>(&self, visitor: &mut V) {
        for stmt in &self.stmts {
            stmt.accept_mut(visitor);
        }
    }

    pub fn flatten(self) -> Vec<Stmt> {
        let mut stmts = Vec::new();
        for stmt in self.stmts {
            match stmt {
                Stmt::Seq(seq) => {
                    stmts.extend(seq.flatten());
                }
                _ => {
                    stmts.push(stmt);
                }
            }
        }
        stmts
    }
}

impl std::fmt::Display for Seq {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for stmt in &self.stmts {
            write!(f, "{}\n", stmt)?;
        }
        Ok(())
    }
}

impl Into<Stmt> for Seq {
    fn into(self) -> Stmt {
        Stmt::Seq(self)
    }
}

impl Into<Stmt> for &Seq {
    fn into(self) -> Stmt {
        Stmt::Seq(self.clone())
    }
}
