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
        .contains("Dimension mismatch: expected 2, got 3"));
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
        .contains("Size mismatch: expected 2, got 3"));
}
