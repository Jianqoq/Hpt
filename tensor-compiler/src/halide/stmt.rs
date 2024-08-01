use std::fmt::Display;

use super::{
    alloca_stmt::AllocaStmt,
    assign_stmt::AssignStmt,
    for_stmt::For,
    if_stmt::IfThenElse,
    inplace_store_stmt::{ InplaceAdd, InplaceDiv, InplaceMul, InplaceStore, InplaceSub },
    let_stmt::LetStmt,
    return_stmt::ReturnStmt,
    seq_stmt::Seq,
    store_stmt::StoreStmt,
    traits::{ Accepter, AccepterMut, AccepterMutate, IRMutVisitor, IRMutateVisitor, IRVisitor },
};

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum Stmt {
    LetStmt(LetStmt),
    StoreStmt(StoreStmt),
    AssignStmt(AssignStmt),
    AllocaStmt(AllocaStmt),
    For(For),
    Seq(Seq),
    InplaceStore(InplaceStore),
    InplaceAdd(InplaceAdd),
    InplaceSub(InplaceSub),
    InplaceMul(InplaceMul),
    InplaceDiv(InplaceDiv),
    IfThenElse(IfThenElse),
    Return(ReturnStmt),
    None,
}

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum StmtType {
    LetStmt,
    AllocaStmt,
    StoreStmt,
    AssignStmt,
    For,
    Seq,
    IfThenElse,
    InplaceStore,
    InplaceAdd,
    InplaceSub,
    InplaceMul,
    InplaceDiv,
    Return,
    None,
}

macro_rules! cast_stmt {
    ($fn_name:ident, $t:ident) => {
        pub fn $fn_name(&self) -> Option<&$t> {
            match self {
                Stmt::$t(e) => Some(e),
                _ => None,
            }
        }
    };
}

macro_rules! cast_stmt_mut {
    ($fn_name:ident, $t:ident) => {
        pub fn $fn_name(&mut self) -> Option<&mut $t> {
            match self {
                Stmt::$t(e) => Some(e),
                _ => None,
            }
        }
    };
}

impl Stmt {
    pub fn is_none(&self) -> bool {
        matches!(self, Stmt::None)
    }

    pub fn same_as(&self, other: &Stmt) -> bool {
        match (self, other) {
            (Stmt::LetStmt(a), Stmt::LetStmt(b)) => { a.var() == b.var() && a.body() == b.body() }
            (Stmt::StoreStmt(a), Stmt::StoreStmt(b)) => {
                a.var() == b.var() &&
                    a.begins() == b.begins() &&
                    a.axes() == b.axes() &&
                    a.steps() == b.steps() &&
                    a.val() == b.val()
            }
            (Stmt::For(a), Stmt::For(b)) => {
                a.var() == b.var() &&
                    a.start() == b.start() &&
                    a.end() == b.end() &&
                    a.stmt() == b.stmt()
            }
            (Stmt::Seq(a), Stmt::Seq(b)) =>
                a
                    .stmts()
                    .iter()
                    .zip(b.stmts().iter())
                    .all(|(a, b)| a.same_as(b)),
            (Stmt::None, Stmt::None) => true,
            _ => false,
        }
    }
    cast_stmt!(to_let_stmt, LetStmt);
    cast_stmt!(to_store_stmt, StoreStmt);
    cast_stmt!(to_for_stmt, For);
    cast_stmt!(to_seq_stmt, Seq);
    cast_stmt_mut!(to_let_stmt_mut, LetStmt);
    cast_stmt_mut!(to_store_stmt_mut, StoreStmt);
    cast_stmt_mut!(to_for_stmt_mut, For);
    cast_stmt_mut!(to_seq_stmt_mut, Seq);

    pub const fn type_info(&self) -> StmtType {
        match self {
            Stmt::LetStmt(_) => StmtType::LetStmt,
            Stmt::StoreStmt(_) => StmtType::StoreStmt,
            Stmt::For(_) => StmtType::For,
            Stmt::Seq(_) => StmtType::Seq,
            Stmt::InplaceStore(_) => StmtType::InplaceStore,
            Stmt::IfThenElse(_) => StmtType::IfThenElse,
            Stmt::InplaceAdd(_) => StmtType::InplaceAdd,
            Stmt::InplaceSub(_) => StmtType::InplaceSub,
            Stmt::InplaceMul(_) => StmtType::InplaceMul,
            Stmt::InplaceDiv(_) => StmtType::InplaceDiv,
            Stmt::AssignStmt(_) => StmtType::AssignStmt,
            Stmt::Return(_) => StmtType::Return,
            Stmt::AllocaStmt(_) => StmtType::AllocaStmt,
            Stmt::None => StmtType::None,
        }
    }
}

impl Display for Stmt {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Stmt::LetStmt(stmt) => write!(f, "{}", stmt),
            Stmt::StoreStmt(stmt) => write!(f, "{}", stmt),
            Stmt::For(stmt) => write!(f, "{}", stmt),
            Stmt::Seq(stmt) => write!(f, "{}", stmt),
            Stmt::InplaceStore(stmt) => write!(f, "{}", stmt),
            Stmt::IfThenElse(stmt) => write!(f, "{}", stmt),
            Stmt::InplaceAdd(stmt) => write!(f, "{}", stmt),
            Stmt::InplaceSub(stmt) => write!(f, "{}", stmt),
            Stmt::InplaceMul(stmt) => write!(f, "{}", stmt),
            Stmt::InplaceDiv(stmt) => write!(f, "{}", stmt),
            Stmt::AssignStmt(stmt) => write!(f, "{}", stmt),
            Stmt::Return(stmt) => write!(f, "{}", stmt),
            Stmt::AllocaStmt(stmt) => write!(f, "{}", stmt),
            Stmt::None => Ok(()),
        }
    }
}

impl Accepter for Stmt {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        match self {
            Stmt::LetStmt(stmt) => {
                stmt.accept(visitor);
            }
            Stmt::StoreStmt(stmt) => {
                stmt.accept(visitor);
            }
            Stmt::For(stmt) => {
                stmt.accept(visitor);
            }
            Stmt::Seq(stmts) => {
                stmts.accept(visitor);
            }
            Stmt::IfThenElse(stmt) => {
                stmt.accept(visitor);
            }
            Stmt::InplaceStore(stmt) => {
                stmt.accept(visitor);
            }
            Stmt::InplaceAdd(stmt) => {
                stmt.accept(visitor);
            }
            Stmt::InplaceSub(stmt) => {
                stmt.accept(visitor);
            }
            Stmt::InplaceMul(stmt) => {
                stmt.accept(visitor);
            }
            Stmt::InplaceDiv(stmt) => {
                stmt.accept(visitor);
            }
            Stmt::AssignStmt(stmt) => {
                stmt.accept(visitor);
            }
            Stmt::Return(stmt) => {
                stmt.accept(visitor);
            }
            Self::AllocaStmt(stmt) => {
                stmt.accept(visitor);
            }
            Stmt::None => {}
        }
    }
}

impl AccepterMut for Stmt {
    fn accept_mut<V: IRMutVisitor>(&self, visitor: &mut V) {
        match self {
            Stmt::LetStmt(stmt) => {
                stmt.accept_mut(visitor);
            }
            Stmt::StoreStmt(stmt) => {
                stmt.accept_mut(visitor);
            }
            Stmt::For(stmt) => {
                stmt.accept_mut(visitor);
            }
            Stmt::Seq(stmts) => {
                stmts.accept_mut(visitor);
            }
            Stmt::IfThenElse(stmt) => {
                stmt.accept_mut(visitor);
            }
            Stmt::InplaceStore(stmt) => {
                stmt.accept_mut(visitor);
            }
            Stmt::InplaceAdd(stmt) => {
                stmt.accept_mut(visitor);
            }
            Stmt::InplaceSub(stmt) => {
                stmt.accept_mut(visitor);
            }
            Stmt::InplaceMul(stmt) => {
                stmt.accept_mut(visitor);
            }
            Stmt::InplaceDiv(stmt) => {
                stmt.accept_mut(visitor);
            }
            Stmt::AssignStmt(stmt) => {
                stmt.accept_mut(visitor);
            }
            Stmt::Return(stmt) => {
                stmt.accept_mut(visitor);
            }
            Stmt::AllocaStmt(stmt) => {
                stmt.accept_mut(visitor);
            }
            Stmt::None => {}
        }
    }
}

impl AccepterMutate for Stmt {
    fn accept_mutate<V: IRMutateVisitor>(&self, visitor: &mut V) {
        match self {
            Stmt::LetStmt(stmt) => {
                stmt.accept_mutate(visitor);
            }
            Stmt::StoreStmt(stmt) => {
                stmt.accept_mutate(visitor);
            }
            Stmt::For(stmt) => {
                stmt.accept_mutate(visitor);
            }
            Stmt::Seq(stmts) => {
                stmts.accept_mutate(visitor);
            }
            Stmt::IfThenElse(stmt) => {
                stmt.accept_mutate(visitor);
            }
            Stmt::InplaceStore(stmt) => {
                stmt.accept_mutate(visitor);
            }
            Stmt::InplaceAdd(stmt) => {
                stmt.accept_mutate(visitor);
            }
            Stmt::InplaceSub(stmt) => {
                stmt.accept_mutate(visitor);
            }
            Stmt::InplaceMul(stmt) => {
                stmt.accept_mutate(visitor);
            }
            Stmt::InplaceDiv(stmt) => {
                stmt.accept_mutate(visitor);
            }
            Stmt::AssignStmt(stmt) => {
                stmt.accept_mutate(visitor);
            }
            Stmt::Return(stmt) => {
                stmt.accept_mutate(visitor);
            }
            Stmt::AllocaStmt(stmt) => {
                stmt.accept_mutate(visitor);
            }
            Stmt::None => {}
        }
    }
}

impl Into<Stmt> for &Stmt {
    fn into(self) -> Stmt {
        self.clone()
    }
}
