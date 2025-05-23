use super::parse::{ Parse, ParseArgs };

pub(crate) struct SqueezeArgs<'a> {
    pub(crate) data: &'a str,
    pub(crate) axes: Option<&'a str>,
}
    
impl<'a> Parse<'a> for SqueezeArgs<'a> {
    fn parse<'b: 'a>(node: &'b crate::onnx::NodeProto) -> SqueezeArgs<'a> {
        let data = node.input[0].as_str();
        let axes = node.input.get(1).map(|s| s.as_str());
        SqueezeArgs {
            data,
            axes,
        }
    }
}
