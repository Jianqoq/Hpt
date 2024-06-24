use serde::{Deserialize, Serialize};



#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub struct BlockInfo {
    pub(crate) current_id: usize,
    pub(crate) parent_id: usize,
}

impl BlockInfo {
    pub fn new(current_id: usize, parent_id: usize) -> Self {
        Self {
            current_id,
            parent_id,
        }
    }

    pub fn parent_id(&self) -> usize {
        self.parent_id
    }

    pub fn current_id(&self) -> usize {
        self.current_id
    }
}

impl Default for BlockInfo {
    fn default() -> Self {
        Self {
            current_id: 0,
            parent_id: 0,
        }
    }
}