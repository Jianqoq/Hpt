use super::parse::{ Parse, ParseArgs };

pub(crate) struct AffineGridArgs<'a> {
    pub(crate) theta: &'a str,
    pub(crate) size: &'a str,
    pub(crate) align_corners: bool,
}

// impl<'a> Parse<AffineGridArgs<'a>> for ParseArgs {
//     type Output<'b> = AffineGridArgs<'b> where Self: 'b;
//     fn parse<'b>(&'b mut self, node: &'b crate::onnx::NodeProto) -> AffineGridArgs<'b> {
//         let theta = node.input[0].as_str();
//         let size = node.input[1].as_str();
//         let align_corners = self.parse_int_attribute(node, "align_corners", 0) == 1;
//         AffineGridArgs {
//             theta,
//             size,
//             align_corners,
//         }
//     }
// }
