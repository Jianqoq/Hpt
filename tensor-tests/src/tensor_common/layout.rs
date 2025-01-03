#![allow(unused_imports)]

use tensor_common::{layout::Layout, shape::shape::Shape, strides::Strides};

#[test]
fn test_reshape() {
    let layout = Layout::new([2, 5, 10], &[50, 10, 1]);
    let a = layout.inplace_reshape(&Shape::from([5, 2, 10])).unwrap();
    assert_eq!(a.shape().inner(), &[5, 2, 10]);
}

#[test]
fn test_reshape_err() {
    let layout = Layout::new([2, 5, 10], &[50, 10, 1]);
    assert!(layout
        .inplace_reshape(&Shape::from([5, 2, 11]))
        .unwrap_err()
        .to_string()
        .contains("Inplace reshape error: Inplace reshape is not possible"));
}

#[test]
fn test_broadcast() {
    let a = Layout::from(&Shape::from([5, 2, 10]));
    let b = Layout::from(&Shape::from([5, 1, 10]));
    let c = a.broadcast(&b).unwrap();
    assert_eq!(c.shape().inner(), &[5, 2, 10]);
}

#[test]
fn test_broadcast_err() {
    let a = Layout::from(&Shape::from([5, 2, 10]));
    let b = Layout::from(&Shape::from([5, 1, 11]));
    assert!(a
        .broadcast(&b)
        .unwrap_err()
        .to_string()
        .contains("Broadcasting error: broadcast failed at index 2, lhs shape: [5, 2, 10], rhs shape: [5, 1, 11]"));
}

#[test]
fn test_layout_from() {
    let a = Layout::from(&Shape::from([5, 2, 10]));
    assert_eq!(a.shape().inner(), &[5, 2, 10]);
    let a = Layout::from(Shape::from([5, 2, 10]));
    assert_eq!(a.shape().inner(), &[5, 2, 10]);
    let a = Layout::from((Shape::from([5, 2, 10]), Strides::from(&[50, 10, 1])));
    assert_eq!(a.shape().inner(), &[5, 2, 10]);
    assert_eq!(a.strides().inner(), &[50, 10, 1]);
    let a = Layout::from((&Shape::from([5, 2, 10]), &Strides::from(&[50, 10, 1])));
    assert_eq!(a.shape().inner(), &[5, 2, 10]);
    assert_eq!(a.strides().inner(), &[50, 10, 1]);
    let a = Layout::from((Shape::from([5, 2, 10]), vec![50, 10, 1]));
    assert_eq!(a.shape().inner(), &[5, 2, 10]);
    assert_eq!(a.strides().inner(), &[50, 10, 1]);
    let a = Layout::from((&Shape::from([5, 2, 10]), vec![50, 10, 1]));
    assert_eq!(a.shape().inner(), &[5, 2, 10]);
    assert_eq!(a.strides().inner(), &[50, 10, 1]);
    let a = Layout::from((&Shape::from([5, 2, 10]), vec![50, 10, 1].as_slice()));
    assert_eq!(a.shape().inner(), &[5, 2, 10]);
    assert_eq!(a.strides().inner(), &[50, 10, 1]);
    let a = Layout::from(&(Shape::from([5, 2, 10]), Strides::from(&[50, 10, 1])));
    assert_eq!(a.shape().inner(), &[5, 2, 10]);
    assert_eq!(a.strides().inner(), &[50, 10, 1]);
    let a = Layout::from(&(Shape::from([5, 2, 10]), Strides::from(&[50, 10, 1])));
    let b = Layout::from(&a);
    assert_eq!(b.shape().inner(), &[5, 2, 10]);
    assert_eq!(b.strides().inner(), &[50, 10, 1]);
}
