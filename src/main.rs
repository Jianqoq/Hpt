

fn main() {
    let a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let strides = [5, 1];

    for i in 0..5 {
        for j in 0..2 {
            print!("{}, ", a[i * strides[1] + j * strides[0]]);
        }
        println!();
    }
}
