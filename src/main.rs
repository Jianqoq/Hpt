// use hpt::{set_cuda_seed, Cuda, NormalReduce, Random, Tensor, TensorError};

#[derive(Debug)]
struct Config {
    block_dim: (usize, usize, usize),
    grid_dim: (usize, usize, usize),
}

fn last_power_of_two(val: usize) -> usize {
    if val <= 1 {
        return val;
    }

    // Find the position of the highest bit set
    // This is equivalent to floor(log2(size))
    let highest_bit = usize::BITS - (val - 1).leading_zeros() - 1;

    // 2^highest_bit is the largest power of 2 <= size
    1 << highest_bit
}


fn main() -> anyhow::Result<()> {
    // set_cuda_seed(1000);
    // let shape = [1024, 1024, 128];
    // let a = Tensor::<f32, Cuda>::randn(&shape)?;
    // let now = std::time::Instant::now();
    // for _ in 0..1 {
    //     let _ = a.sum([0, 1], true)?;
    // }
    // println!("{:?}", now.elapsed() / 1);
    // let sum = a.sum([0, 1], true)?;
    // println!("{}", sum);
    // let sum_cpu = a.to_cpu::<0>()?.sum([0, 1], true)?;
    // println!("{}", sum_cpu);
    let compute_cfg = |output_size: usize, fast_dim_size: usize, reduce_size: usize, input_size: usize| {
        let mut cfg = Config {
            block_dim: (0, 0, 0),
            grid_dim: (0, 0, 0),
        };
        let max_el_per_thread = 512;
        let min_el_per_thread = 16;
        let mut max_block_size = 512usize;
        let block_x = if fast_dim_size < 32 { last_power_of_two(fast_dim_size as usize) } else { 32 };
        max_block_size /= block_x;
        let block_y = if reduce_size < max_block_size {
            last_power_of_two(reduce_size)
        } else {
            max_block_size
        };
        let total_threads = block_x * block_y;
        assert!(total_threads <= 512);
        let num_el_per_output = input_size / output_size;
        let curr_num_el_per_thread = num_el_per_output / block_y;
        let adjusted_el_per_thread = curr_num_el_per_thread
            .min(max_el_per_thread)
            .max(min_el_per_thread);
        let grid_y = (num_el_per_output / block_y / adjusted_el_per_thread).min(65536);
        let grid_x = output_size / block_x;
        cfg.block_dim = (block_x, block_y, 1);
        cfg.grid_dim = (grid_x, grid_y, 1);
        // check_launch_config(a.device(), &cfg).unwrap();
        cfg
    };
    let shape = [1024, 1024, 6];
    let size = shape.iter().product::<usize>();
    let axes = [0, 1];
    let reduce_size = shape.iter().enumerate().filter(|(i, _)| axes.contains(&i)).map(|(_, &d)| d).product::<usize>();
    println!("reduce_size: {}", reduce_size);
    let output_size = size / reduce_size;
    let fast_dim_size = shape[shape.len() - 1];
    let cfg = compute_cfg(output_size, fast_dim_size, reduce_size, size);
    println!("{:#?}", cfg);
    
    let num_threads_to_reduce = cfg.block_dim.1 * cfg.grid_dim.1;
    println!("num_threads_to_reduce: {}", num_threads_to_reduce);
    let num_el_per_thread = reduce_size / num_threads_to_reduce;
    println!("num_el_per_thread: {}", num_el_per_thread);
    Ok(())
}
