use petgraph::{dot::{Config, Dot}, prelude::StableGraph};

#[allow(unused)]
pub(crate) fn generate_online_graphviz_link<N, E>(graph: &StableGraph<N, E>) -> String
    where N: std::fmt::Debug, E: std::fmt::Debug
{
    let dot = Dot::with_config(&graph, &[Config::EdgeNoLabel]);
    let dot_string = format!("{:?}", dot);

    // URL编码DOT字符串
    let encoded = urlencoding::encode(&dot_string);
    format!("https://dreampuf.github.io/GraphvizOnline/#{encoded}")
}
