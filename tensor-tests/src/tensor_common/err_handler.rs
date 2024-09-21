#![allow(unused_imports)]
use tensor_common::err_handler::ErrHandler;

#[test]
fn test_check_ndim_match() {
    ErrHandler::check_ndim_match(2, 2).unwrap();
}

#[test]
fn test_check_ndim_match_err() {
    assert_eq!(
        ErrHandler::check_ndim_match(2, 3).unwrap_err().to_string(),
        r"expect ndim to be 3 but got 2, at tensor-tests\src\tensor_common\err_handler.rs:12:9"
    );
}

#[test]
fn test_check_same_axis() {
    ErrHandler::check_same_axis(1, 2).unwrap();
}

#[test]
fn test_check_same_axis_err() {
    assert_eq!(
        ErrHandler::check_same_axis(1, 1).unwrap_err().to_string(),
        r"axis should be unique, but got 1 and 1, at tensor-tests\src\tensor_common\err_handler.rs:25:9"
    );
}

#[test]
fn test_check_index_in_range() {
    ErrHandler::check_index_in_range(2, 1).unwrap();
    ErrHandler::check_index_in_range(2, -1).unwrap();
    let mut index = 1;
    ErrHandler::check_index_in_range_mut(2, &mut index).unwrap();
    assert_eq!(index, 1);
    let mut index = -1;
    ErrHandler::check_index_in_range_mut(2, &mut index).unwrap();
    assert_eq!(index, 1);
}

#[test]
fn test_check_index_in_range_err() {
    assert_eq!(
        ErrHandler::check_index_in_range(2, 2).unwrap_err().to_string(),
        r"tensor ndim is 2 but got index `2`, at tensor-tests\src\tensor_common\err_handler.rs:45:9"
    );
    assert_eq!(
        ErrHandler::check_index_in_range(2, -3).unwrap_err().to_string(),
        r"tensor ndim is 2 but got converted index from `-3` to `-1`, at tensor-tests\src\tensor_common\err_handler.rs:49:9"
    );
    assert_eq!(
        ErrHandler::check_index_in_range_mut(2, &mut 2).unwrap_err().to_string(),
        r"tensor ndim is 2 but got index `2`, at tensor-tests\src\tensor_common\err_handler.rs:53:9"
    );
    assert_eq!(
        ErrHandler::check_index_in_range_mut(2, &mut -3).unwrap_err().to_string(),
        r"tensor ndim is 2 but got converted index from `-3` to `-1`, at tensor-tests\src\tensor_common\err_handler.rs:57:9"
    );
}

#[test]
fn test_size_match() {
    ErrHandler::check_size_match(2, 2).unwrap();
}

#[test]
fn test_size_match_err() {
    assert_eq!(
        ErrHandler::check_size_match(2, 3).unwrap_err().to_string(),
        r"expect size 2 but got size 3, at tensor-tests\src\tensor_common\err_handler.rs:70:9"
    );
}
