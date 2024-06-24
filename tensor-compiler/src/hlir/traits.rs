pub trait HlirAcceptor {
    fn accept<V: HlirVisitor>(&self, visitor: &V);
}

pub trait HlirAccepterMut {
    fn accept_mut<V: HlirMutVisitor>(&self, visitor: &mut V);
}

pub trait HlirAccepterMutate {
    fn accept_mutate<V: HlirMutateVisitor>(&self, visitor: &mut V);
}

pub trait HlirVisitor {}

pub trait HlirMutVisitor {}

pub trait HlirMutateVisitor {}
