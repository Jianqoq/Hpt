use front_end::{ context::Context, tensor::Tensor };
use halide::r#type::{ HalideirTypeCode, Type };
use serde_json::json;

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
    pub mod node;
    pub mod traits;
}

pub mod front_end {
    pub mod context;
    pub mod graph;
    pub mod tensor;
    pub mod _tensor;
    pub mod std_ops;
    pub mod control_flow;
}

pub mod op;
pub mod float;

static I64_TYPE: Type = Type::new(HalideirTypeCode::Int, 64, 1);

pub fn visualize<const C: usize, T: FnMut(&Context) -> [Tensor; C]>(mut f: T) -> serde_json::Value {
    let mut context = Context::new();
    let _ = f(&mut context);
    json!(context.ctx().as_ref().clone())
}
