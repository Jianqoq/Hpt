use getset::CopyGetters;
use getset::Getters;
use getset::MutGetters;
use getset::Setters;
use hashbrown::HashMap;
use hashbrown::HashSet;
use serde::Deserialize;
use serde::Serialize;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum BlockType {
    If,
    While,
    For,
    Function,
    Loop,
    Placeholder,
}

#[derive(Getters, Setters, MutGetters, CopyGetters, Debug, Clone, Serialize, Deserialize)]
pub struct BlockManager {
    #[getset(get = "pub", set = "pub", get_mut = "pub")]
    block_nodes: HashMap<usize, HashSet<usize>>,
    #[getset(get = "pub", set = "pub", get_mut = "pub")]
    block_parent: HashMap<usize, usize>,
    #[getset(get = "pub", set = "pub", get_mut = "pub")]
    block_parents: HashMap<usize, HashSet<usize>>,
    #[getset(get = "pub", set = "pub", get_mut = "pub")]
    block_children: HashMap<usize, HashSet<usize>>,
    #[getset(get = "pub", set = "pub", get_mut = "pub")]
    block_type: HashMap<usize, BlockType>,
    #[getset(get = "pub", set = "pub", get_mut = "pub")]
    block_names: HashMap<usize, String>,
}

impl BlockManager {
    pub fn insert_parent(
        &mut self,
        block_id: usize,
        block_name: String,
        parent_block_id: usize,
        block_type: BlockType,
    ) {
        self.block_parent.insert(block_id, parent_block_id);
        self.block_nodes
            .entry(parent_block_id)
            .or_insert(HashSet::new())
            .insert(block_id);
        let parents = get_parents(block_id, self);
        self.block_parents.insert(block_id, parents);
        self.block_children
            .entry(parent_block_id)
            .or_insert(HashSet::new())
            .insert(block_id);
        self.block_type.insert(block_id, block_type);
        self.block_names.insert(block_id, block_name);
    }

    pub fn remove_block(&mut self, block_id: &usize) -> bool {
        if let Some(block_type) = self.block_type.get(block_id) {
            if *block_type != BlockType::Function {
                return false;
            }
        }
        let block_id = *block_id;
        if let Some(parent_block_id) = self.block_parent.get(&block_id).cloned() {
            self.block_children.get(&block_id).map(|children| {
                for child in children.iter() {
                    self.block_parent.get_mut(child).map(|parent| {
                        *parent = parent_block_id;
                    });
                    self.block_parents.get_mut(child).map(|parents| {
                        parents.remove(&block_id);
                        parents.insert(parent_block_id);
                    });
                }
            });
            // remove the block from the parent's children block
            if let Some(block_children) = self.block_children.get(&block_id).cloned() {
                self.block_children
                    .get_mut(&parent_block_id)
                    .map(|children| {
                        children.remove(&block_id);
                        children.extend(block_children);
                    });
            }
        }

        self.block_parents.iter_mut().for_each(|(_, value)| {
            value.remove(&block_id);
        });
        self.block_parents.remove(&block_id);
        self.block_children.remove(&block_id);
        self.block_parent.remove(&block_id);
        self.block_names.remove(&block_id);
        true
    }

    pub fn flatten_block(&mut self) {
        let vec = self.block_parent.keys().cloned().collect::<Vec<usize>>();
        for id in vec {
            self.remove_block(&id);
        }
        self.remove_block(&0);
    }

    pub fn new() -> Self {
        Default::default()
    }
}

fn get_parents(block_id: usize, block_manager: &BlockManager) -> HashSet<usize> {
    let parent_block_id = block_manager.block_parent.get(&block_id).map(|x| *x);
    let mut parents = HashSet::new();
    if let Some(mut parent_block_id) = parent_block_id {
        parents.insert(parent_block_id);
        while let Some(parent) = block_manager.block_parent.get(&parent_block_id) {
            parent_block_id = *parent;
            parents.insert(parent_block_id);
        }
    }
    parents
}

impl Default for BlockManager {
    fn default() -> Self {
        BlockManager {
            block_nodes: HashMap::new(),
            block_parent: HashMap::new(),
            block_parents: HashMap::new(),
            block_children: HashMap::new(),
            block_type: HashMap::new(),
            block_names: HashMap::new(),
        }
    }
}
