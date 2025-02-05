use criterion::Criterion;

mod benchmarks;

fn main() {
    #[cfg(any(feature = "cos", feature = "unary"))]
    benchmarks::unary::unary_benches::cos_benches();
    #[cfg(any(feature = "sin", feature = "unary"))]
    benchmarks::unary::unary_benches::sin_benches();
    #[cfg(any(feature = "selu", feature = "unary"))]
    benchmarks::unary::unary_benches::selu_benches();
    #[cfg(any(feature = "tan", feature = "unary"))]
    benchmarks::unary::unary_benches::tan_benches();
    #[cfg(any(feature = "acos", feature = "unary"))]
    benchmarks::unary::unary_benches::acos_benches();
    #[cfg(any(feature = "asin", feature = "unary"))]
    benchmarks::unary::unary_benches::asin_benches();
    #[cfg(any(feature = "atan", feature = "unary"))]
    benchmarks::unary::unary_benches::atan_benches();
    #[cfg(any(feature = "cosh", feature = "unary"))]
    benchmarks::unary::unary_benches::cosh_benches();
    #[cfg(any(feature = "sinh", feature = "unary"))]
    benchmarks::unary::unary_benches::sinh_benches();
    #[cfg(any(feature = "tanh", feature = "unary"))]
    benchmarks::unary::unary_benches::tanh_benches();
    #[cfg(any(feature = "acosh", feature = "unary"))]
    benchmarks::unary::unary_benches::acosh_benches();
    #[cfg(any(feature = "asinh", feature = "unary"))]
    benchmarks::unary::unary_benches::asinh_benches();
    #[cfg(any(feature = "atanh", feature = "unary"))]
    benchmarks::unary::unary_benches::atanh_benches();
    #[cfg(any(feature = "sigmoid", feature = "unary"))]
    benchmarks::unary::unary_benches::sigmoid_benches();
    #[cfg(any(feature = "exp", feature = "unary"))]
    benchmarks::unary::unary_benches::exp_benches();
    #[cfg(any(feature = "relu", feature = "unary"))]
    benchmarks::unary::unary_benches::relu_benches();
    #[cfg(any(feature = "mish", feature = "unary"))]
    benchmarks::unary::unary_benches::mish_benches();
    #[cfg(any(feature = "softplus", feature = "unary"))]
    benchmarks::unary::unary_benches::softplus_benches();
    #[cfg(any(feature = "relu6", feature = "unary"))]
    benchmarks::unary::unary_benches::relu6_benches();
    #[cfg(any(feature = "hard_sigmoid", feature = "unary"))]
    benchmarks::unary::unary_benches::hard_sigmoid_benches();
    #[cfg(any(feature = "gelu", feature = "unary"))]
    benchmarks::unary::unary_benches::gelu_benches();
    #[cfg(any(feature = "leaky_relu", feature = "unary"))]
    benchmarks::unary::unary_benches::leaky_relu_benches();
    #[cfg(any(feature = "elu", feature = "unary"))]
    benchmarks::unary::unary_benches::elu_benches();
    #[cfg(any(feature = "celu", feature = "unary"))]
    benchmarks::unary::unary_benches::celu_benches();
    #[cfg(any(feature = "log10", feature = "unary"))]
    benchmarks::unary::unary_benches::log10_benches();
    #[cfg(any(feature = "log2", feature = "unary"))]
    benchmarks::unary::unary_benches::log2_benches();
    #[cfg(any(feature = "recip", feature = "unary"))]
    benchmarks::unary::unary_benches::recip_benches();
    #[cfg(any(feature = "exp2", feature = "unary"))]
    benchmarks::unary::unary_benches::exp2_benches();
    #[cfg(any(feature = "sqrt", feature = "unary"))]
    benchmarks::unary::unary_benches::sqrt_benches();
    #[cfg(any(feature = "neg", feature = "unary"))]
    benchmarks::unary::unary_benches::neg_benches();
    #[cfg(any(feature = "square", feature = "unary"))]
    benchmarks::unary::unary_benches::square_benches();
    #[cfg(any(feature = "abs", feature = "unary"))]
    benchmarks::unary::unary_benches::abs_benches();
    #[cfg(any(feature = "ceil", feature = "unary"))]
    benchmarks::unary::unary_benches::ceil_benches();
    #[cfg(any(feature = "sign", feature = "unary"))]
    benchmarks::unary::unary_benches::sign_benches();
    #[cfg(any(feature = "clip", feature = "unary"))]
    benchmarks::unary::unary_benches::clip_benches();

    #[cfg(feature = "softmax")]
    benchmarks::softmax::softmax::softmax_benches();

    #[cfg(feature = "cat")]
    benchmarks::shape_manipulate::concat::cat_benches();
    #[cfg(any(feature = "sum", feature = "reduction"))]
    benchmarks::reduction::reduction_benches::sum_benches();
    #[cfg(any(feature = "prod", feature = "reduction"))]
    benchmarks::reduction::reduction_benches::prod_benches();

    #[cfg(all(feature = "f32", feature = "conv2d"))]
    benchmarks::conv::conv2d::conv2d_benches();
    #[cfg(any(feature = "f32", feature = "maxpool"))]
    benchmarks::conv::maxpool::maxpool_benches();

    #[cfg(feature = "hamming")]
    benchmarks::signals::hamming_window::benches();

    Criterion::default().configure_from_args().final_summary();
}
