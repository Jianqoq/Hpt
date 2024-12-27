/// this macro is used to implement the FloatOutBinaryPromote trait for a given type
#[macro_export]
macro_rules! impl_float_out_binary_promote {
    ($lhs:ty, $rhs:ty, $output:ty) => {
        impl FloatOutBinaryPromote<$rhs> for $lhs {
            type Output = $output;
        }
        impl FloatOutBinaryPromote<Scalar<$rhs>> for Scalar<$lhs> {
            type Output = Scalar<$output>;
        }
        paste::paste! {
            impl FloatOutBinaryPromote<[<$rhs _promote>]> for [<$lhs _promote>] {
                type Output = [<$output _promote>];
            }
        }
    };
}

/// this macro is used to implement the FloatOutBinaryPromote trait for a given type
#[macro_export]
macro_rules! impl_normal_out_promote {
    ($lhs:ty, $rhs:ty, $output:ty) => {
        impl NormalOutPromote<$rhs> for $lhs {
            type Output = $output;
        }
        impl NormalOutPromote<Scalar<$rhs>> for Scalar<$lhs> {
            type Output = Scalar<$output>;
        }
        paste::paste! {
            impl NormalOutPromote<[<$rhs _promote>]> for [<$lhs _promote>] {
                type Output = [<$output _promote>];
            }
        }
    };
}

/// this macro is used to implement the SimdCmpPromote trait for a given type
#[macro_export]
macro_rules! impl_simd_cmp_promote {
    ($lhs:ty, $rhs:ty, $output:ty) => {
        paste::paste! {
            impl SimdCmpPromote<[<$rhs _promote>]> for [<$lhs _promote>] {
                type Output = [<$output _promote>];
            }
        }
    };
}

/// this macro is used to implement the FloatOutUnaryPromote trait for a given type
#[macro_export]
macro_rules! impl_float_out_unary_promote {
    ($lhs:ty, $output:ty) => {
        impl FloatOutUnaryPromote for $lhs {
            type Output = $output;
        }
        impl FloatOutUnaryPromote for Scalar<$lhs> {
            type Output = Scalar<$output>;
        }
        paste::paste! {
            impl FloatOutUnaryPromote for [<$lhs _promote>] {
                type Output = [<$output _promote>];
            }
        }
    };
}

