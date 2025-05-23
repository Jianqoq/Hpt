use crate::onnx::NodeProto;

pub(crate) struct ParseArgs {
    arg_idx: usize,
}

impl ParseArgs {
    pub(crate) fn new() -> Self {
        Self { arg_idx: 0 }
    }

    pub(crate) fn parse_int_attribute(
        &mut self,
        node: &NodeProto,
        target: &str,
        default: i64
    ) -> i64 {
        if let Some(attr) = node.attribute.get(self.arg_idx) {
            if attr.name() == target {
                let res = attr.i.unwrap_or(default);
                self.arg_idx += 1;
                res
            } else {
                default
            }
        } else {
            default
        }
    }
}

pub(crate) trait Parse<'p> {
    fn parse<'a: 'p>(node: &'a NodeProto) -> Self;
}
