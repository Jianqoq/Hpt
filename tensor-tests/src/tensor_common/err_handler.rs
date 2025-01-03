#![allow(unused_imports)]
use tensor_common::error::shape::ShapeError;

#[test]
fn test_check_ndim_match() {
    ShapeError::check_dim(2, 2).unwrap();
}

#[test]
fn test_check_ndim_match_err() {
    assert!(ShapeError::check_dim(2, 3)
        .unwrap_err()
        .to_string()
        .contains("expect ndim to be 3 but got 2"));
}

#[test]
fn test_size_match() {
    ShapeError::check_size_match(2, 2).unwrap();
}

#[test]
fn test_size_match_err() {
    assert!(ShapeError::check_size_match(2, 3)
        .unwrap_err()
        .to_string()
        .contains("expect size 2 but got size 3"));
}

