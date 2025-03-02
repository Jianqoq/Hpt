pub(crate) fn rearrange_array(ndim: usize, to_reduce: &[usize]) -> Vec<usize> {
    let mut origin_order = (0..ndim).collect::<Vec<usize>>();
    let mut to_reduce = to_reduce.to_vec();
    // sort the reduce axes
    to_reduce.sort();

    // store the elements to be reduced
    let mut moved_elements = Vec::new();
    origin_order.retain(|&x| {
        if to_reduce.contains(&x) {
            moved_elements.push(x);
            false
        } else {
            true
        }
    });

    // put the reduced elements at the end
    origin_order.extend(moved_elements);

    origin_order
}

pub(crate) fn is_keep_fast_dim(strides: &[i64], axes: &[usize]) -> bool {
    let mut keep_fast_dim = true;
    for axis in axes.iter() {
        if strides[*axis] == 1 {
            keep_fast_dim = false;
            break;
        }
    }
    keep_fast_dim
}
