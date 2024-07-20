use std::{ cell::RefCell, rc::Rc };

#[derive(Clone)]
pub struct RcMut<T> {
    inner: Rc<RefCell<T>>,
}

impl<T> RcMut<T> {
    pub fn new(inner: T) -> Self {
        RcMut {
            inner: Rc::new(RefCell::new(inner)),
        }
    }

    pub fn borrow(&self) -> std::cell::Ref<T> {
        self.inner.borrow()
    }

    pub fn borrow_mut(&self) -> std::cell::RefMut<T> {
        self.inner.borrow_mut()
    }
}
