#![allow(unused_imports)]

use hpt_common::{
    shape::shape::Shape,
    shape::shape_utils::{
        compare_and_pad_shapes, get_broadcast_axes_from, mt_intervals_simd, yield_one_after,
        yield_one_before,
    },
};

#[test]
fn test_basic_division_mt_intervals_simd() {
    let intervals = mt_intervals_simd(100, 4, 10);
    assert_eq!(intervals.len(), 4);
    assert_eq!(intervals[0], (0, 30));
    assert_eq!(intervals[1], (30, 60));
    assert_eq!(intervals[2], (60, 80));
    assert_eq!(intervals[3], (80, 100));
}

#[test]
fn test_more_threads_than_work_mt_intervals_simd() {
    let intervals = mt_intervals_simd(16, 10, 4);
    assert_eq!(intervals.len(), 10);
    assert_eq!(intervals[0], (0, 4));
    assert_eq!(intervals[1], (4, 8));
    assert_eq!(intervals[2], (8, 12));
    assert_eq!(intervals[3], (12, 16));
    for i in 4..10 {
        assert_eq!(intervals[i], (16, 16));
    }
}

#[test]
fn test_vec_size_gt_outter_loop_size_mt_intervals_simd() {
    let intervals = mt_intervals_simd(4, 10, 8);
    assert_eq!(intervals.len(), 10);
    assert_eq!(intervals[0], (0, 4));
    for i in 1..10 {
        assert_eq!(intervals[i], (0, 0));
    }
}

#[test]
fn test_vec_size_not_dividing_outer_loop_size_mt_intervals_simd() {
    let intervals = mt_intervals_simd(102, 4, 10);
    assert_eq!(intervals.len(), 4);
    assert_eq!(intervals[0], (0, 30));
    assert_eq!(intervals[1], (30, 60));
    assert_eq!(intervals[2], (60, 80));
    assert_eq!(intervals[3], (80, 102));
}

#[test]
fn test_yield_one_after() {
    let shape = Shape::from(&[1, 2, 3]);
    let new = yield_one_after(&shape, 1);
    assert_eq!(&new, &[1, 2, 1, 3]);
}

#[test]
fn test_yield_one_before() {
    let shape = Shape::from(&[1, 2, 3]);
    let new = yield_one_before(&shape, 1);
    assert_eq!(&new, &[1, 1, 2, 3]);
}

#[test]
fn test_compare_and_pad_shapes() {
    let shape1 = Shape::from(&[1, 2, 1, 3]);
    let shape2 = Shape::from(&[1, 3, 2]);
    let (a, b) = compare_and_pad_shapes(&shape1, &shape2);
    assert_eq!(&a, &[1, 2, 1, 3]);
    assert_eq!(&b, &[1, 1, 3, 2]);
    let (a, b) = compare_and_pad_shapes(&shape2, &shape1);
    assert_eq!(&a, &[1, 2, 1, 3]);
    assert_eq!(&b, &[1, 1, 3, 2]);
}

#[test]
fn test_get_broadcast_axes_from() {
    let shape1 = Shape::from(&[1, 2, 1, 3]);
    let res_shape = Shape::from(&[1, 1, 3, 2]);
    let axes = get_broadcast_axes_from(&shape1, &res_shape);
    match axes {
        Ok(_) => panic!("Should return Err"),
        Err(err) => {
            assert!(err.to_string().contains("Broadcasting error: broadcast failed at index 1, lhs shape: [1, 2, 1, 3], rhs shape: [1, 1, 3, 2]"));
        }
    }
}

#[test]
#[should_panic(expected = "vec_size must be greater than zero")]
fn test_zero_vec_size() {
    mt_intervals_simd(100, 4, 0);
}
