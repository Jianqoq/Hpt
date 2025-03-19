pub(crate) fn generate_all_combinations(arr: &[usize]) -> Vec<Vec<i64>> {
    let n = arr.len();
    let total_combinations = 1 << n;
    let mut result = Vec::with_capacity(total_combinations);

    for i in 0..total_combinations {
        let mut combination = Vec::new();
        for j in 0..n {
            if (i & (1 << j)) != 0 {
                combination.push(arr[j] as i64);
            }
        }
        if combination.len() > 0 {
            result.push(combination);
        }
    }

    result
}
