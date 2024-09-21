#![allow(unused_imports)]

use tensor_common::shape_utils::mt_intervals_simd;

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
#[should_panic(expected = "vec_size must be greater than zero")]
fn test_zero_vec_size() {
    mt_intervals_simd(100, 4, 0);
}
