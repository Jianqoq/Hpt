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

pub(crate) fn get_fast_dim_size(shape: &[i64], strides: &[i64], axes: &[usize]) -> i64 {
    for axis in axes.iter() {
        if strides[*axis] == 1 {
            return shape[*axis];
        }
    }
    unreachable!()
}

pub(crate) fn split_groups_by_axes(groups: &Vec<Vec<usize>>, axes: &[usize]) -> Vec<Vec<usize>> {
    let mut result = Vec::new();

    for group in groups {
        let mut current_group = Vec::new();
        let mut current_is_reduce = false;
        let mut first_dim = true;

        for &dim in group {
            let is_reduce = axes.contains(&dim);

            if first_dim {
                current_is_reduce = is_reduce;
                current_group.push(dim);
                first_dim = false;
            } else if is_reduce == current_is_reduce {
                current_group.push(dim);
            } else {
                result.push(current_group);
                current_group = vec![dim];
                current_is_reduce = is_reduce;
            }
        }

        if !current_group.is_empty() {
            result.push(current_group);
        }
    }

    result
}

pub(crate) fn get_new_reduce_axes(groups: Vec<Vec<usize>>, axes: &[usize]) -> Vec<usize> {
    let mut result = vec![];
    for (idx, group) in groups.into_iter().enumerate() {
        if group.iter().any(|&x| axes.contains(&x)) {
            let all_reduce = group.iter().all(|&x| axes.contains(&x));
            if !all_reduce {
                panic!("Inconsistent reduction properties in dimension group");
            }
            result.push(idx);
        }
    }
    result
}

pub(crate) fn get_new_shape(groups: &Vec<Vec<usize>>, shape: &[i64]) -> Vec<i64> {
    let mut result = vec![];
    for group in groups {
        result.push(group.iter().map(|&x| shape[x]).product::<i64>());
    }
    result
}
