fn conv2d_grouped(
    input: &Vec<Vec<Vec<f32>>>,
    kernels: &Vec<Vec<Vec<Vec<f32>>>>,
    groups: usize,
    stride: usize,
    padding: usize,
    dilation: usize
) -> Vec<Vec<Vec<f32>>> {
    let in_channels = input.len();
    let in_height = input[0].len();
    let in_width = input[0][0].len();
    let num_kernels = kernels.len();
    let kernel_height = kernels[0][0].len();
    let kernel_width = kernels[0][0][0].len();

    let in_channels_per_group = in_channels / groups;
    let kernels_per_group = num_kernels / groups;

    let output_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    let output_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    let mut output = vec![vec![vec![0.0; output_width]; output_height]; num_kernels];

    for g in 0..groups {
        for k in 0..kernels_per_group {
            for c in 0..in_channels_per_group {
                
                for y in 0..output_height {
                    for x in 0..output_width {
                        let mut sum = 0.0;
                        for i in 0..kernel_height {
                            for j in 0..kernel_width {
                                let in_y = y * stride + i * dilation - padding;
                                let in_x = x * stride + j * dilation - padding;
                                if in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width {
                                    let in_val = input[g * in_channels_per_group + c][in_y as usize][in_x as usize];
                                    let kernel_val = kernels[g * kernels_per_group + k][c][i][j];
                                    sum += in_val * kernel_val;
                                }
                            }
                        }
                        output[g * kernels_per_group + k][y][x] += sum;
                    }
                }
            }
        }
    }

    output
}

fn main() {
    // Example usage
    let input = vec![
        vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![6.0, 7.0, 8.0, 9.0, 10.0],
            vec![8.0, 9.0, 10.0, 11.0, 12.0]
        ]
    ]; // Single-channel input
    let kernels = vec![vec![vec![vec![1.0, 2.0, 3.0]; 3]; 1]; 2]; // Two kernels, single channel each
    let groups = 1;
    let stride = 2;
    let padding = 2;
    let dilation = 2;

    let output = conv2d_grouped(&input, &kernels, groups, stride, padding, dilation);
    for i in 0..output.len() {
        for j in 0..output[0].len() {
            for k in 0..output[0][0].len() {
                print!("{} ", output[i][j][k]);
            }
            println!();
        }
        println!();
    }
}
