#![allow(unused_imports)]

use std::sync::Arc;

use tensor_common::strides::Strides;

#[test]
fn test_inner() {
    let strides = Strides::from(&[1, 2, 3]);
    assert_eq!(strides.inner(), &[1, 2, 3]);
}

#[test]
fn test_to_string() {
    let strides = Strides::from(&[1, 2, 3]);
    let string = format!("{:?}", strides);
    assert_eq!(string, "strides([1, 2, 3])");
}

#[test]
fn test_from() {
    let strides = Strides::from(Arc::new(vec![1, 2, 3]));
    assert_eq!(strides.inner(), &[1, 2, 3]);
    let strides = Strides::from([1, 2, 3]);
    assert_eq!(strides.inner(), &[1, 2, 3]);
    let vec = &vec![1, 2, 3];
    let strides = Strides::from(vec);
    assert_eq!(strides.inner(), &[1, 2, 3]);
    let vec = &Arc::new(vec![1, 2, 3]);
    let strides = Strides::from(vec);
    assert_eq!(strides.inner(), &[1, 2, 3]);
    let vec = Arc::new([1i64, 2, 3]);
    let strides = Strides::from(vec);
    assert_eq!(strides.inner(), &[1, 2, 3]);
}

#[test]
fn test_default() {
    let strides = Strides::default();
    let arr: [i64; 0] = [];
    assert_eq!(strides.inner(), &arr);
}

#[test]
fn test_dref_mut() {
    let mut strides = Strides::from(&[1, 2, 3]);
    assert_eq!(strides.inner(), &[1, 2, 3]);
    *&mut strides = Strides::from(&[3, 2, 1]);
    assert_eq!(strides.inner(), &[3, 2, 1]);
}
