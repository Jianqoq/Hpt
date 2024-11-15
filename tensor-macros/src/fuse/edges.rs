use std::ops::{ Deref, DerefMut };

use std::collections::{ HashMap, HashSet };

#[derive(Default, Debug, Clone)]
pub(crate) struct Edges<'ast> {
    inner: HashMap<&'ast syn::Ident, HashSet<&'ast syn::Ident>>,
}

impl<'ast> Edges<'ast> {
    pub fn invert(&'ast self) -> Edges<'ast> {
        let mut inverted: HashMap<&'ast syn::Ident, HashSet<&'ast syn::Ident>> = HashMap::new();
        for (&key, value) in self.inner.iter() {
            for i in value {
                if let Some(set) = inverted.get_mut(i) {
                    set.insert(key);
                } else {
                    let mut set = HashSet::new();
                    set.insert(key);
                    inverted.insert(i, set);
                }
            }
        }
        Edges { inner: inverted }
    }
}

impl<'ast> From<HashMap<&'ast syn::Ident, HashSet<&'ast syn::Ident>>> for Edges<'ast> {
    fn from(inner: HashMap<&'ast syn::Ident, HashSet<&'ast syn::Ident>>) -> Self {
        Edges { inner }
    }
}

impl<'ast> Deref for Edges<'ast> {
    type Target = HashMap<&'ast syn::Ident, HashSet<&'ast syn::Ident>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'ast> DerefMut for Edges<'ast> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
