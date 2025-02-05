#![allow(unused_imports)]

use hpt_common::{layout::layout::Layout, shape::shape::Shape, utils::pointer::Pointer};

#[test]
fn test_index() {
    let mut a = [10, 11, 12, 13];
    let mut ptr = Pointer::new(&mut a as *mut i32, 4);
    assert_eq!(ptr[0usize], 10);
    ptr += 1i64;
    assert_eq!(ptr[0usize], 11);
    ptr += 1isize;
    assert_eq!(ptr[0usize], 12);
    ptr += 1usize;
    assert_eq!(ptr[0usize], 13);
    ptr -= 1i64;
    assert_eq!(ptr[0usize], 12);
    ptr -= 1isize;
    assert_eq!(ptr[0usize], 11);
    ptr -= 1usize;
    assert_eq!(ptr[0usize], 10);

    ptr += 1i64;
    assert_eq!(*ptr, 11);
    *ptr = 20;
    assert_eq!(*ptr, 20);

    let string = format!("{}", ptr);
    assert_eq!(
        string,
        format!("Pointer( ptr: {}, val: {} )", ptr.ptr as usize, ptr[0usize])
    );
}
