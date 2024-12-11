macro_rules! poly {
    // POLY2: c1*x + c0
    ($x:expr, $c1:expr, $c0:expr) => {
        $x.mul_add($c1, $c0)
    };
    // POLY3: c2*x^2 + c1*x + c0
    ($x:expr, $x2:expr, $c2:expr, $c1:expr, $c0:expr) => {
            $x2.mul_add($c2, $x.mul_add($c1, $c0))
    };
    // POLY4: c3*x^3 + c2*x^2 + c1*x + c0
    ($x:expr, $x2:expr, $c3:expr, $c2:expr, $c1:expr, $c0:expr) => {
            $x2.mul_add(
                $x.mul_add($c3, $c2),
                $x.mul_add($c1, $c0)
            )
    };
    // POLY5: c4*x^4 + c3*x^3 + c2*x^2 + c1*x + c0
    ($x:expr, $x2:expr, $x4:expr, $c4:expr, $c3:expr, $c2:expr, $c1:expr, $c0:expr) => {
            $x4.mul_add(
                $c4,
                poly!($x, $x2, $c3, $c2, $c1, $c0)
            )
    };
    // POLY6: c5*x^5 + c4*x^4 + c3*x^3 + c2*x^2 + c1*x + c0
    ($x:expr, $x2:expr, $x4:expr, $c5:expr, $c4:expr, $c3:expr, $c2:expr, $c1:expr, $c0:expr) => {
            $x4.mul_add(
                poly!($x, $c5, $c4),
                poly!($x, $x2, $c3, $c2, $c1, $c0)
            )
    };

    // POLY7
    (
        $x:expr,
        $x2:expr,
        $x4:expr,
        $c6:expr,
        $c5:expr,
        $c4:expr,
        $c3:expr,
        $c2:expr,
        $c1:expr,
        $c0:expr
    ) => {
            $x4.mul_add(
                poly!($x, $x2, $c6, $c5, $c4),
                poly!($x, $x2, $c3, $c2, $c1, $c0)
            )
    };

    // POLY8
    (
        $x:expr,
        $x2:expr,
        $x4:expr,
        $c7:expr,
        $c6:expr,
        $c5:expr,
        $c4:expr,
        $c3:expr,
        $c2:expr,
        $c1:expr,
        $c0:expr
    ) => {
            $x4.mul_add(
                poly!($x, $x2, $c7, $c6, $c5, $c4),
                poly!($x, $x2, $c3, $c2, $c1, $c0)
            )
    };
}
