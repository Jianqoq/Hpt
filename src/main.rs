use tensor_dyn::{ set_global_display_precision, TensorCreator };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

fn max_pool2d_dilated(
    input: &Vec<Vec<Vec<Vec<f32>>>>, // 输入数据: batches, channels, height, width
    kernel_size: (usize, usize),     // 池化窗口大小: (height, width)
    stride: usize,                   // 步长
    padding: usize,                  // 填充
    dilation: usize                  // 扩张
) -> Vec<Vec<Vec<Vec<f32>>>> {
    let (kernel_height, kernel_width) = kernel_size;

    // 计算输出尺寸
    let batches = input.len();
    let channels = input[0].len();
    let in_height = input[0][0].len();
    let in_width = input[0][0][0].len();

    let output_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    let output_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    // 初始化输出向量
    let mut output = vec![vec![vec![vec![0.0f32; output_width]; output_height]; channels]; batches];

    for b in 0..batches {
        for c in 0..channels {
            for y in 0..output_height {
                for x in 0..output_width {
                    let mut max_val = f32::MIN;  // 初始化最小值，使用浮点数的最小值

                    // 遍历每个池化窗口
                    for i in 0..kernel_height {
                        for j in 0..kernel_width {
                            let in_y = y * stride + i * dilation - padding;
                            let in_x = x * stride + j * dilation - padding;

                            // 检查边界条件
                            if in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width {
                                let val = input[b][c][in_y as usize][in_x as usize];
                                if val > max_val {
                                    max_val = val;
                                }
                            }
                        }
                    }

                    // 将找到的最大值赋给输出
                    output[b][c][y][x] = max_val;
                }
            }
        }
    }

    output
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    set_global_display_precision(7);
    let kernel = _Tensor::<f32>::new([
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
        ],
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
        ],
    ]);
    let a = _Tensor::<f32>::arange(0, 100)?.reshape([2, 2, 5, 5])?;

    let res = a.conv(
        &kernel,
        None,
        None,
        Some(&[2, 2]),
        Some(
            &[
                (2, 2),
                (2, 2),
            ]
        ),
        Some(&[2, 2]),
        Some(2)
    )?;

    let res = a.img2col(
        kernel.shape(),
        Some(&[2, 2]),
        Some(
            &[
                (2, 2),
                (2, 2),
            ]
        ),
        Some(&[2, 2]),
        Some(2)
    )?;
    println!("{:?}", res.reshape([2, 3, 3, 3, 9])?);
    Ok(())
}
