#![allow(unused_imports)]
use tensor_common::axis::{process_axes, Axis};

#[test]
#[should_panic = "Error message is correct"]
fn test_out_of_range() {
    match process_axes(10, 2) {
        Ok(_) => panic!("Should panic"),
        Err(e) => {
            if e.to_string()
                .contains("Dimension out of range: expected in 0..2, got 10")
            {
                panic!("Error message is correct");
            } else {
                panic!("Error message is incorrect");
            }
        }
    }
}

#[test]
#[should_panic = "Error message is correct"]
fn test_out_of_range_cvt() {
    match process_axes(-3, 2) {
        Ok(_) => panic!("Should panic"),
        Err(e) => {
            if e.to_string()
                .contains("Dimension out of range: expected in 0..2, got -1")
            {
                panic!("Error message is correct");
            } else {
                panic!("Error message is incorrect");
            }
        }
    }
}

#[test]
fn test_out_of_range_normal() {
    match process_axes(-1, 2) {
        Ok(a) => {
            assert_eq!(a[0], 1);
            assert_eq!(a.len(), 1);
        }
        Err(e) => {
            panic!("Should not panic, but got error: {}", e);
        }
    }
}

#[test]
fn test_from() {
    let a: Axis = Axis::from(&[1i64, 2, 3]);
    assert_eq!(a.axes, &[1, 2, 3]);
    let a: Axis = Axis::from(&[1i32, 2, 3]);
    assert_eq!(a.axes, &[1, 2, 3]);
    let a: Axis = Axis::from(&[1i16, 2, 3]);
    assert_eq!(a.axes, &[1, 2, 3]);
    let a: Axis = Axis::from(&[1i8, 2, 3]);
    assert_eq!(a.axes, &[1, 2, 3]);
    let a: Axis = Axis::from(&[1i128, 2, 3]);
    assert_eq!(a.axes, &[1, 2, 3]);
    let a: Axis = Axis::from(&[1usize, 2, 3]);
    assert_eq!(a.axes, &[1, 2, 3]);
    let a: Axis = Axis::from(&[1isize, 2, 3]);
    assert_eq!(a.axes, &[1, 2, 3]);

    let a: Axis = Axis::from([1i64, 2, 3]);
    assert_eq!(a.axes, &[1, 2, 3]);
    let a: Axis = Axis::from([1i32, 2, 3]);
    assert_eq!(a.axes, &[1, 2, 3]);
    let a: Axis = Axis::from([1i16, 2, 3]);
    assert_eq!(a.axes, &[1, 2, 3]);
    let a: Axis = Axis::from([1i8, 2, 3]);
    assert_eq!(a.axes, &[1, 2, 3]);
    let a: Axis = Axis::from([1i128, 2, 3]);
    assert_eq!(a.axes, &[1, 2, 3]);
    let a: Axis = Axis::from([1usize, 2, 3]);
    assert_eq!(a.axes, &[1, 2, 3]);
    let a: Axis = Axis::from([1isize, 2, 3]);
    assert_eq!(a.axes, &[1, 2, 3]);

    let vec = vec![1i64, 2, 3];
    let vec_ref = &vec;
    let a: Axis = Axis::from(vec_ref);
    assert_eq!(a.axes, &[1, 2, 3]);
    let slice = vec.as_slice();
    let a: Axis = Axis::from(slice);
    assert_eq!(a.axes, &[1, 2, 3]);

    let vec = vec![1i32, 2, 3];
    let vec_ref = &vec;
    let a: Axis = Axis::from(vec_ref);
    assert_eq!(a.axes, &[1, 2, 3]);
    let slice = vec.as_slice();
    let a: Axis = Axis::from(slice);
    assert_eq!(a.axes, &[1, 2, 3]);

    let vec = vec![1i16, 2, 3];
    let vec_ref = &vec;
    let a: Axis = Axis::from(vec_ref);
    assert_eq!(a.axes, &[1, 2, 3]);
    let slice = vec.as_slice();
    let a: Axis = Axis::from(slice);
    assert_eq!(a.axes, &[1, 2, 3]);

    let vec = vec![1i8, 2, 3];
    let vec_ref = &vec;
    let a: Axis = Axis::from(vec_ref);
    assert_eq!(a.axes, &[1, 2, 3]);
    let slice = vec.as_slice();
    let a: Axis = Axis::from(slice);
    assert_eq!(a.axes, &[1, 2, 3]);

    let vec = vec![1i128, 2, 3];
    let vec_ref = &vec;
    let a: Axis = Axis::from(vec_ref);
    assert_eq!(a.axes, &[1, 2, 3]);
    let slice = vec.as_slice();
    let a: Axis = Axis::from(slice);
    assert_eq!(a.axes, &[1, 2, 3]);

    let vec = vec![1usize, 2, 3];
    let vec_ref = &vec;
    let a: Axis = Axis::from(vec_ref);
    assert_eq!(a.axes, &[1, 2, 3]);
    let slice = vec.as_slice();
    let a: Axis = Axis::from(slice);
    assert_eq!(a.axes, &[1, 2, 3]);

    let vec = vec![1isize, 2, 3];
    let vec_ref = &vec;
    let a: Axis = Axis::from(vec_ref);
    assert_eq!(a.axes, &[1, 2, 3]);
    let slice = vec.as_slice();
    let a: Axis = Axis::from(slice);
    assert_eq!(a.axes, &[1, 2, 3]);

    let a: Axis = Axis::from(1i64);
    assert_eq!(a.axes, &[1]);
    let a: Axis = Axis::from(1i32);
    assert_eq!(a.axes, &[1]);
    let a: Axis = Axis::from(1i16);
    assert_eq!(a.axes, &[1]);
    let a: Axis = Axis::from(1i8);
    assert_eq!(a.axes, &[1]);
    let a: Axis = Axis::from(1i128);
    assert_eq!(a.axes, &[1]);
    let a: Axis = Axis::from(1usize);
    assert_eq!(a.axes, &[1]);
    let a: Axis = Axis::from(1isize);
    assert_eq!(a.axes, &[1]);
    let a: Axis = Axis::from(&1i64);
    assert_eq!(a.axes, &[1]);
    let a: Axis = Axis::from(&1i32);
    assert_eq!(a.axes, &[1]);
    let a: Axis = Axis::from(&1i16);
    assert_eq!(a.axes, &[1]);
    let a: Axis = Axis::from(&1i8);
    assert_eq!(a.axes, &[1]);
    let a: Axis = Axis::from(&1i128);
    assert_eq!(a.axes, &[1]);
    let a: Axis = Axis::from(&1usize);
    assert_eq!(a.axes, &[1]);
    let a: Axis = Axis::from(&1isize);
    assert_eq!(a.axes, &[1]);
}
