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
    #[cfg(feature = "cat")]
    benchmarks::shape_manipulate::concat::cat_benches();
    #[cfg(any(feature = "sum", feature = "reduction"))]
    benchmarks::reduction::reduction_benches::sum_benches();
    #[cfg(any(feature = "prod", feature = "reduction"))]
    benchmarks::reduction::reduction_benches::prod_benches();

    Criterion::default().configure_from_args().final_summary();
}
