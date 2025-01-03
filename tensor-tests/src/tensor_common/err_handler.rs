#![allow(unused_imports)]
use tensor_common::err_handler::TensorError;

#[test]
fn test_check_ndim_match() {
    TensorError::check_ndim_match(2, 2).unwrap();
}

#[test]
fn test_check_ndim_match_err() {
    assert!(TensorError::check_ndim_match(2, 3)
        .unwrap_err()
        .to_string()
        .contains("expect ndim to be 3 but got 2"));
}

#[test]
fn test_check_same_axis() {
    TensorError::check_same_axis(1, 2).unwrap();
}

#[test]
fn test_check_same_axis_err() {
    assert!(TensorError::check_same_axis(1, 1)
        .unwrap_err()
        .to_string()
        .contains("axis should be unique, but got 1 and 1"));
}

#[test]
fn test_check_index_in_range() {
    TensorError::check_index_in_range(2, 1).unwrap();
    TensorError::check_index_in_range(2, -1).unwrap();
    let mut index = 1;
    TensorError::check_index_in_range_mut(2, &mut index).unwrap();
    assert_eq!(index, 1);
    let mut index = -1;
    TensorError::check_index_in_range_mut(2, &mut index).unwrap();
    assert_eq!(index, 1);
}

#[test]
fn test_check_index_in_range_err() {
    assert!(TensorError::check_index_in_range(2, 2)
        .unwrap_err()
        .to_string()
        .contains("tensor ndim is 2 but got index `2`"));
    assert!(TensorError::check_index_in_range(2, -3)
        .unwrap_err()
        .to_string()
        .contains("tensor ndim is 2 but got converted index from `-3` to `-1`"));
    assert!(TensorError::check_index_in_range_mut(2, &mut 2)
        .unwrap_err()
        .to_string()
        .contains("tensor ndim is 2 but got index `2`"));
    assert!(TensorError::check_index_in_range_mut(2, &mut -3)
        .unwrap_err()
        .to_string()
        .contains("tensor ndim is 2 but got converted index from `-3` to `-1`"));
}

#[test]
fn test_size_match() {
    TensorError::check_size_match(2, 2).unwrap();
}

#[test]
fn test_size_match_err() {
    assert!(TensorError::check_size_match(2, 3)
        .unwrap_err()
        .to_string()
        .contains("expect size 2 but got size 3"));
}
