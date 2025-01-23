use crate::utils::pointer::Pointer;

/// Update the loop progress
#[inline]
pub fn next<const N: usize, T>(
    prg: &mut [i64],
    main_shape: &[i64],
    mut ptrs: [&mut Pointer<T>; N],
    shapes: [&[i64]; N],
    strides: [&[i64]; N]
) {
    let dim = shapes[0].len();
    for j in (0..dim - 1).rev() {
        let j = j;
        if prg[j] < main_shape[j] {
            prg[j] += 1;
            for i in 0..N {
                ptrs[i] += strides[i][j];
            }
            break;
        } else {
            prg[j] = 0;
            for i in 0..N {
                ptrs[i] -= strides[i][j] * shapes[i][j];
            }
        }
    }
}

/// Update the loop progress where shape need to be subtracted by 1
#[inline]
pub fn next_sub1<const N: usize, T>(
    prg: &mut [i64],
    main_shape: &[i64],
    mut ptrs: [&mut Pointer<T>; N],
    shapes: [&[i64]; N],
    strides: [&[i64]; N]
) {
    let dim = shapes[0].len();
    for j in (0..dim - 1).rev() {
        let j = j;
        if prg[j] < main_shape[j] - 1 {
            prg[j] += 1;
            for i in 0..N {
                ptrs[i] += strides[i][j];
            }
            break;
        } else {
            prg[j] = 0;
            for i in 0..N {
                ptrs[i] -= strides[i][j] * (shapes[i][j] - 1);
            }
        }
    }
}
