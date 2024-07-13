use std::{ hash::Hash, ops::{ Deref, DerefMut } };

use hashbrown::{ HashMap, HashSet };
use serde::Serialize;

#[derive(Default, Debug, Clone, Serialize)]
pub struct Edges<T> where T: Hash + Eq {
    inner: HashMap<T, HashSet<T>>,
}

impl<T> Edges<T> where T: Hash + Eq + Clone {
    pub fn new() -> Self {
        Edges {
            inner: HashMap::new(),
        }
    }

    pub fn inner(&self) -> &HashMap<T, HashSet<T>> {
        &self.inner
    }

    pub fn invert(&self) -> Edges<T> {
        let mut inverted: HashMap<T, HashSet<T>> = HashMap::new();
        for (key, value) in self.inner.iter() {
            for i in value.iter() {
                if let Some(set) = inverted.get_mut(i) {
                    set.insert(key.clone());
                } else {
                    let mut set = HashSet::new();
                    set.insert(key.clone());
                    inverted.insert(i.clone(), set);
                }
            }
        }
        Edges { inner: inverted }
    }
    pub fn set_inner(&mut self, inner: HashMap<T, HashSet<T>>) {
        self.inner = inner;
    }
}

impl<T> From<HashMap<T, HashSet<T>>> for Edges<T> where T: Hash + Eq {
    fn from(inner: HashMap<T, HashSet<T>>) -> Self {
        Edges { inner }
    }
}

impl<T> Deref for Edges<T> where T: Hash + Eq {
    type Target = HashMap<T, HashSet<T>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> DerefMut for Edges<T> where T: Hash + Eq {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
