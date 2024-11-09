use std::ops::{ Deref, DerefMut };

use std::collections::{ HashMap, HashSet };

use super::dag::Var;

#[derive(Default, Debug, Clone)]
pub(crate) struct Edges<'ast> {
    inner: HashMap<Var<'ast>, HashSet<syn::Ident>>,
}

impl<'ast> Edges<'ast> {
    pub fn invert(&'ast self) -> Edges<'ast> {
        let mut inverted: HashMap<Var<'ast>, HashSet<syn::Ident>> = HashMap::new();
        for (key, value) in self.inner.iter() {
            for i in value {
                if let Some(set) = inverted.get_mut(&(Var { ident: i })) {
                    set.insert(key.ident.clone());
                } else {
                    let mut set = HashSet::new();
                    set.insert(key.ident.clone());
                    inverted.insert(Var { ident: i }, set);
                }
            }
        }
        Edges { inner: inverted }
    }
}

impl<'ast> From<HashMap<Var<'ast>, HashSet<syn::Ident>>> for Edges<'ast> {
    fn from(inner: HashMap<Var<'ast>, HashSet<syn::Ident>>) -> Self {
        Edges { inner }
    }
}

impl<'ast> Deref for Edges<'ast> {
    type Target = HashMap<Var<'ast>, HashSet<syn::Ident>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'ast> DerefMut for Edges<'ast> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
