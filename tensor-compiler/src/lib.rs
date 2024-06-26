
use halide::r#type::{ HalideirTypeCode, Type };

pub mod halide {
    pub mod expr;
    pub mod for_stmt;
    pub mod let_stmt;
    pub mod substitute {
        pub mod subsititue_var;
        pub mod subsititue_expr;
        pub mod subsititue_for_range;
        pub mod substitue_load_partial_idx;
    }
    pub mod utils {
        pub mod for_utils;
    }
    pub mod if_stmt;
    pub mod inplace_store_stmt;
    pub mod r#type;
    pub mod ir_cmp;
    pub mod printer;
    pub mod seq_stmt;
    pub mod stmt;
    pub mod store_stmt;
    pub mod tests;
    pub mod traits;
    pub mod exprs;
    pub mod variable;
}

pub mod hlir {
    pub mod _value;
    pub mod expr;
    pub mod traits;
    pub mod exprs;
    pub mod printer;
    pub mod func_type;
    pub mod tests;
}

pub mod op;
pub mod float;

static I64_TYPE: Type = Type::new(HalideirTypeCode::Int, 64, 1);
