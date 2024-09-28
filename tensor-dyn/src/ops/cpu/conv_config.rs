use crate::CONV_REGNUM;
use tensor_traits::CommonBounds;
use tensor_types::traits::{Init, VecCommon, VecTrait};

/// An enumeration representing different algorithms for determining kernel parameters.
///
/// This enum is used to select the algorithm that determines kernel block sizes for
/// convolutional operations or other tensor-related computations. It provides two strategies:
/// `Heuristic` and `Greedy`.
#[derive(Debug, Clone, Copy, Default)]
pub enum KernelParamAlgo {
    /// Uses a heuristic-based approach to determine kernel parameters.
    Heuristic,
    /// Uses a greedy algorithm to optimize kernel parameters based on
    /// available information like cache size and tensor shape.
    #[default]
    Greedy,
}

/// Configuration for 2D convolution operations.
///
/// This structure holds configuration parameters for a 2D convolution operation,
/// including the number of input and output channels, kernel size, and block sizes
/// for optimization. It also takes into account the L1 cache size for efficient
/// memory usage during the convolution process.
#[derive(Debug, Clone)]
pub struct Conv2dConfig<T> {
    /// The size of the L1 cache in terms of number of elements of type `T`.
    pub(crate) l1_cache_size: usize,
    /// The number of output channels for the convolution operation.
    pub(crate) out_channels: i64,
    /// The number of input channels for the convolution operation.
    pub(crate) in_channels: i64,
    /// The block size for the output channels.
    pub(crate) co_block_size: i64,
    /// The block size for the input channels.
    pub(crate) ci_block_size: i64,
    /// The kernel size for the convolution operation.
    pub(crate) kernel_size: [i64; 2],
    /// Phantom data to hold the type `T`.
    pub(crate) _phantom: std::marker::PhantomData<T>,
}

impl<T> Conv2dConfig<T>
where
    T: CommonBounds,
{
    /// Sets the L1 cache size (divided by the size of `T`).
    pub fn set_l1_cache_size(&mut self, l1_cache_size: usize) -> &mut Self {
        self.l1_cache_size = l1_cache_size / size_of::<T>();
        self
    }
    /// Sets the block size for the output channels.
    pub fn set_co_block_size(&mut self, co_block_size: i64) -> &mut Self {
        self.co_block_size = co_block_size;
        self
    }
    /// Sets the block size for the input channels.
    pub fn set_ci_block_size(&mut self, ci_block_size: i64) -> &mut Self {
        self.ci_block_size = ci_block_size;
        self
    }
    /// Creates a new `Conv2dConfig` with the specified output channels, input channels, and kernel size.
    pub fn new(
        out_channels: i64,
        in_channels: i64,
        kernel_size: [i64; 2],
        algo: KernelParamAlgo,
    ) -> Self {
        let cache_size = cache_size::l1_cache_size().unwrap_or(128 * 1024) / size_of::<T>();
        let (co_block_size, ci_block_size) = match algo {
            KernelParamAlgo::Heuristic => todo!(),
            KernelParamAlgo::Greedy => find_exact_combination::<T, CONV_REGNUM>(
                cache_size as i64,
                out_channels,
                in_channels,
                kernel_size[1],
                kernel_size[0],
            ),
        };
        Self {
            l1_cache_size: cache_size::l1_cache_size().unwrap_or(128 * 1024) / size_of::<T>(),
            out_channels,
            in_channels,
            kernel_size,
            _phantom: std::marker::PhantomData,
            co_block_size,
            ci_block_size,
        }
    }
    /// Returns the number of elements that can fit in the L1 cache.
    pub fn l1_cache_size(&self) -> usize {
        self.l1_cache_size * size_of::<T>()
    }
    /// Returns the output channels size
    pub fn out_channels(&self) -> i64 {
        self.out_channels
    }
    /// Returns the input channels size
    pub fn in_channels(&self) -> i64 {
        self.in_channels
    }
    /// Returns the kernel size
    pub fn kernel_size(&self) -> [i64; 2] {
        self.kernel_size
    }
    /// Sets the output channels size
    pub fn set_out_channels(&mut self, out_channels: i64) -> &mut Self {
        self.out_channels = out_channels;
        self
    }
    /// Sets the input channels size
    pub fn set_in_channels(&mut self, in_channels: i64) -> &mut Self {
        self.in_channels = in_channels;
        self
    }
    /// Sets the kernel size
    pub fn set_kernel_size(&mut self, kernel_size: [i64; 2]) -> &mut Self {
        self.kernel_size = kernel_size;
        self
    }
}

/// a greedy algorithm to find the best combination of co_block_size and ci_block_size
#[allow(unused)]
fn find_exact_combination<T: CommonBounds, const REGNUM: usize>(
    max_cache_size: i64,
    max_co_b: i64,
    max_ci_b: i64,
    weight_size: i64,
    height_size: i64,
) -> (i64, i64)
where
    T::Vec: VecTrait<T> + Copy + Init<T> + VecCommon,
{
    let mut best_co_b = 1;
    let mut best_ci_b = 1;

    for ci_b in (1..max_ci_b + 1).rev() {
        for co_b in (1..max_co_b + 1)
            .rev()
            .filter(|&co_b| (co_b % (T::Vec::SIZE as i64)) * 2 == 0)
        {
            let product = co_b * (REGNUM as i64)
                + weight_size * height_size * ci_b * ((REGNUM as i64) + co_b);

            if product <= max_cache_size && co_b <= max_co_b && ci_b <= max_ci_b {
                if ci_b > best_ci_b || (ci_b == best_ci_b && co_b > best_co_b) {
                    best_co_b = co_b;
                    best_ci_b = ci_b;
                }
            }
        }
    }
    let cache_line_size = (cache_size::cache_line_size(1, cache_size::CacheType::Data)
        .unwrap_or(64)
        / size_of::<T>()) as i64;
    if max_co_b >= cache_line_size && best_co_b < cache_line_size {
        best_co_b = cache_line_size;
        if best_ci_b / 2 != 0 {
            best_ci_b /= 2;
        } else {
            best_ci_b = 1;
        }
    }

    (best_co_b, best_ci_b)
}

#[allow(unused)]
fn find_exact_combination2<T: CommonBounds, const REGNUM: usize>(
    max_cache_size: i64,
    max_co_b: i64,
    max_ci_b: i64,
    weight_size: i64,
    height_size: i64,
) -> (i64, i64)
where
    T::Vec: VecTrait<T> + Copy + Init<T> + VecCommon,
{
    let mut best_co_b = 1;
    let mut best_ci_b = 1;

    for co_b in (1..max_co_b + 1)
        .rev()
        .filter(|&co_b| co_b % (T::Vec::SIZE as i64) == 0)
    {
        for ci_b in (1..max_ci_b + 1).rev() {
            let product = co_b * (REGNUM as i64)
                + weight_size * height_size * ci_b * ((REGNUM as i64) + co_b);

            if product <= max_cache_size && co_b <= max_co_b && ci_b <= max_ci_b {
                if ci_b > best_ci_b || (ci_b == best_ci_b && co_b > best_co_b) {
                    best_co_b = co_b;
                    best_ci_b = ci_b;
                }
            }
        }
    }
    let cache_line_size = (cache_size::cache_line_size(1, cache_size::CacheType::Data)
        .unwrap_or(64)
        / size_of::<T>()) as i64;
    if max_co_b >= cache_line_size && best_co_b < cache_line_size {
        best_co_b = cache_line_size;
        if best_ci_b / 2 != 0 {
            best_ci_b /= 2;
        } else {
            best_ci_b = 1;
        }
    }

    (best_co_b, best_ci_b)
}

/// Configuration for 3D convolution operations.
///
/// This structure holds configuration parameters for a 3D convolution operation,
/// including the number of input and output channels, kernel size, and block sizes
/// for optimization. It also accounts for the L1 cache size to ensure efficient
/// memory usage during the convolution process.
#[derive(Debug, Clone)]
pub struct Conv3dConfig<T> {
    /// The size of the L1 cache, expressed in terms of the number of elements of type `T`.
    pub(crate) l1_cache_size: usize,
    /// The number of output channels for the convolution operation.
    pub(crate) out_channels: i64,
    /// The number of input channels for the convolution operation.
    pub(crate) in_channels: i64,
    /// The block size for the output channels, used for optimizing cache performance.
    pub(crate) co_block_size: i64,
    /// The block size for the input channels, used for optimizing cache performance.
    pub(crate) ci_block_size: i64,
    /// A 3-element array specifying the size of the convolution kernel (depth, height, width).
    pub(crate) kernel_size: [i64; 3],
    /// A phantom data marker to tie the configuration to a specific data type `T`.
    pub(crate) _phantom: std::marker::PhantomData<T>,
}

impl<T> Conv3dConfig<T>
where
    T: CommonBounds,
{
    /// Sets the L1 cache size (divided by the size of `T`).
    pub fn set_l1_cache_size(&mut self, l1_cache_size: usize) -> &mut Self {
        self.l1_cache_size = l1_cache_size / size_of::<T>();
        self
    }
    /// Sets the block size for the output channels.
    pub fn set_co_block_size(&mut self, co_block_size: i64) -> &mut Self {
        self.co_block_size = co_block_size;
        self
    }
    /// Sets the block size for the input channels.
    pub fn set_ci_block_size(&mut self, ci_block_size: i64) -> &mut Self {
        self.ci_block_size = ci_block_size;
        self
    }
    /// Creates a new `Conv3dConfig` with the specified output channels, input channels, and kernel size.
    /// 
    /// # Arguments
    /// 
    /// * `out_channels` - The number of output channels for the convolution operation.
    /// * `in_channels` - The number of input channels for the convolution operation.
    /// * `kernel_size` - A 3-element array specifying the size of the convolution kernel (depth, height, width).
    /// * `algo` - The algorithm to use for determining kernel parameters.
    /// 
    /// # Returns
    /// 
    /// A new `Conv3dConfig` instance with the specified parameters.
    pub fn new(
        out_channels: i64,
        in_channels: i64,
        kernel_size: [i64; 3],
        algo: KernelParamAlgo,
    ) -> Self {
        let cache_size = cache_size::l1_cache_size().unwrap_or(32 * 1024) / size_of::<T>();
        let (co_block_size, ci_block_size) = match algo {
            KernelParamAlgo::Heuristic => todo!(),
            KernelParamAlgo::Greedy => find_exact_combination_3d::<T, CONV_REGNUM>(
                cache_size as i64,
                out_channels,
                in_channels,
                kernel_size[1],
                kernel_size[0],
                kernel_size[2],
            ),
        };
        Self {
            l1_cache_size: cache_size::l1_cache_size().unwrap_or(32 * 1024) / size_of::<T>(),
            out_channels,
            in_channels,
            kernel_size,
            _phantom: std::marker::PhantomData,
            co_block_size,
            ci_block_size,
        }
    }
    /// Returns the number of elements that can fit in the L1 cache.
    pub fn l1_cache_size(&self) -> usize {
        self.l1_cache_size * size_of::<T>()
    }
    /// Returns the number of output channels.
    pub fn out_channels(&self) -> i64 {
        self.out_channels
    }
    /// Returns the number of input channels.
    pub fn in_channels(&self) -> i64 {
        self.in_channels
    }
    /// Returns the kernel size.
    pub fn kernel_size(&self) -> [i64; 3] {
        self.kernel_size
    }
    /// Set the number of output channels.
    pub fn set_out_channels(&mut self, out_channels: i64) -> &mut Self {
        self.out_channels = out_channels;
        self
    }
    /// Set the number of input channels.
    pub fn set_in_channels(&mut self, in_channels: i64) -> &mut Self {
        self.in_channels = in_channels;
        self
    }
    /// Set the kernel size.
    pub fn set_kernel_size(&mut self, kernel_size: [i64; 3]) -> &mut Self {
        self.kernel_size = kernel_size;
        self
    }
}

fn find_exact_combination_3d<T: CommonBounds, const REGNUM: usize>(
    max_cache_size: i64,
    max_co_b: i64,
    max_ci_b: i64,
    weight_size: i64,
    height_size: i64,
    depth_size: i64,
) -> (i64, i64)
where
    T::Vec: VecTrait<T> + Copy + Init<T> + VecCommon,
{
    let mut best_co_b = 0;
    let mut best_ci_b = 0;

    for co_b in (1..=max_co_b)
        .rev()
        .filter(|&co_b| co_b % (T::Vec::SIZE as i64) == 0)
    {
        for ci_b in (1..=max_ci_b).rev() {
            let product = co_b * (REGNUM as i64)
                + weight_size * height_size * depth_size * ci_b * ((REGNUM as i64) + co_b);

            if product <= max_cache_size {
                if co_b > best_co_b || (co_b == best_co_b && ci_b > best_ci_b) {
                    best_co_b = co_b;
                    best_ci_b = ci_b;
                }
            }
        }
    }

    (best_co_b, best_ci_b)
}
