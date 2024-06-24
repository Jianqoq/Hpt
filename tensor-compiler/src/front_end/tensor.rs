#![allow(dead_code)]
use std::{ cell::RefCell, rc::Rc };

use tensor_common::{ err_handler::ErrHandler, layout::Layout };

use crate::op::Op;

pub struct Tensor {
    pub(crate) inputs: Vec<usize>,
    pub(crate) op: Op,
    pub(crate) layout: Layout,
    pub(crate) name: Option<Rc<String>>,
    pub(crate) error_msg: Rc<RefCell<Option<ErrHandler>>>,
    pub(crate) block_id: Rc<RefCell<usize>>,
    pub(crate) id: Rc<RefCell<usize>>,
}
