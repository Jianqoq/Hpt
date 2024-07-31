use std::hint::black_box;
use std::mem;
use std::time::{ Duration, Instant };
use rand::rngs::ThreadRng;
use rand::{ Rng, thread_rng };

fn main() {
    let mut rng = thread_rng();
    let num_iterations = 512 * 1024 * 1024 * 2;
    let mut data: Vec<(i64, i64)> = Vec::with_capacity(num_iterations);
    // 先生成随机数
    for _ in 0..num_iterations {
        let a: i64 = rng.gen();
        let b: i64 = rng.gen();
        data.push((a, b));
    }

    // 测试乘法性能
    let start_mul = Instant::now();
    let mut result = 0;
    for &(a, b) in &data {
        result = a + b;
    }
    let duration_mul = start_mul.elapsed();
    println!("result: {}", result);
    // 测试除法性能
    let start_div = Instant::now();
    for &(a, b) in &data {
        // 避免除以零
        result = a % b
    }
    let duration_div = start_div.elapsed();
    println!("result: {}", result);
    println!("Multiplication took: {:?}, Division took: {:?}", duration_mul, duration_div);
}
