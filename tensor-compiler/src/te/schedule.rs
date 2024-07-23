use std::{ collections::HashMap, fmt::{ self, Display, Formatter } };

use crate::halide::{ printer::IRPrinter, stmt::Stmt };

use super::stages::Stage;

pub struct Schedule {
    pub(crate) map: HashMap<usize, Stage>,
}

impl Schedule {
    pub fn to_halide(&self) -> Stmt {
        let last = self.map
            .iter()
            .map(|(k, _)| k)
            .max()
            .unwrap_or(&0);
        let stage = self.map.get(last).unwrap();
        stage.to_halide(&self.map)
    }
}

impl Display for Schedule {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", IRPrinter.print_stmt_str(self.to_halide()))
    }
}
