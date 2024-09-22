#![allow(unused_imports)]
use std::sync::Arc;

use tensor_common::shape::Shape;

#[test]
fn test_new() {
    let shape = Shape::new(&[1, 2, 3]);
    assert_eq!(shape.inner(), &[1, 2, 3]);
}

#[test]
fn test_to_strides() {
    let shape = Shape::new(&[1, 2, 3]);
    let strides = shape.to_strides();
    assert_eq!(strides.inner(), &[6, 3, 1]);
}

#[test]
fn test_to_string() {
    let shape = Shape::new(&[1, 2, 3]);
    let string = format!("{:?}", shape);
    assert_eq!(string, "shape([1, 2, 3])");
}

#[test]
fn test_default() {
    let shape = Shape::default();
    let arr: [i64; 0] = [];
    assert_eq!(shape.inner(), &arr);
}

#[test]
fn test_from() {
    let shape = Shape::from(&Arc::new(vec![1, 2, 3]));
    assert_eq!(shape.inner(), &[1, 2, 3]);
    let shape = Shape::from(Arc::new([1, 2, 3]));
    assert_eq!(shape.inner(), &[1, 2, 3]);
}