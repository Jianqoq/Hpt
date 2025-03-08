#![allow(unused_imports)]
use hpt::common::cpu::TensorLike;
use hpt::common::TensorInfo;
use hpt::ops::*;
use hpt::utils::*;
use hpt::Tensor;

#[test]
fn test_set_global_display_precision() -> anyhow::Result<()> {
    let a = Tensor::<f32>::new(0.39871234566541987416541651);
    assert_eq!(a.to_string(), "Tensor(0.3987)\n");
    set_display_precision(8);
    assert_eq!(a.to_string(), "Tensor(0.39871234)\n");
    Ok(())
}

#[test]
fn test_set_global_lr_precision() -> anyhow::Result<()> {
    let a = Tensor::<f32>::arange(0, 100)?.reshape(&[10, 10])?;
    assert_eq!(
        a.to_string(),
        "Tensor([[ 0.  1.  2. ...  7.  8.  9. ]
        [10. 11. 12. ... 17. 18. 19. ]
        [20. 21. 22. ... 27. 28. 29. ]
        ...

        [70. 71. 72. ... 77. 78. 79. ]
        [80. 81. 82. ... 87. 88. 89. ]
        [90. 91. 92. ... 97. 98. 99. ]], shape=(10, 10), strides=(10, 1), dtype=f32)\n"
    );
    set_display_elements(2);
    assert_eq!(
        a.to_string(),
        "Tensor([[ 0.  1. ...  8.  9. ]
        [10. 11. ... 18. 19. ]
        ...

        [80. 81. ... 88. 89. ]
        [90. 91. ... 98. 99. ]], shape=(10, 10), strides=(10, 1), dtype=f32)\n"
    );
    Ok(())
}

#[test]
fn test_set_num_threads() -> anyhow::Result<()> {
    // we have two thread pools, one from rayon and one from ThreadPool
    set_num_threads(10);
    assert_eq!(get_num_threads(), 10); // we get the number of threads from ThreadPool
    set_num_threads(16);
    assert_eq!(get_num_threads(), 16); // we get the number of threads from ThreadPool
    Ok(())
}
