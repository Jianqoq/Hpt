use tensor_traits::CommonBounds;
use tensor_types::{ dtype::TypeCommon, traits::{ Init, VecSize, VecTrait } };
use crate::CONV_REGNUM;

#[derive(Debug, Clone, Copy, Default)]
pub enum KernelParamAlgo {
    Heuristic,
    #[default]
    Greedy,
}

#[derive(Debug, Clone)]
pub struct Conv2dConfig<T> {
    pub(crate) l1_cache_size: usize,
    pub(crate) out_channels: i64,
    pub(crate) in_channels: i64,
    pub(crate) co_block_size: i64,
    pub(crate) ci_block_size: i64,
    pub(crate) kernel_size: [i64; 2],
    pub(crate) _phantom: std::marker::PhantomData<T>,
}

impl<T> Conv2dConfig<T> where T: CommonBounds {
    pub fn set_l1_cache_size(&mut self, l1_cache_size: usize) -> &mut Self {
        self.l1_cache_size = l1_cache_size / std::mem::size_of::<T>();
        self
    }
    pub fn set_co_block_size(&mut self, co_block_size: i64) -> &mut Self {
        self.co_block_size = co_block_size;
        self
    }
    pub fn set_ci_block_size(&mut self, ci_block_size: i64) -> &mut Self {
        self.ci_block_size = ci_block_size;
        self
    }
    pub fn new(
        out_channels: i64,
        in_channels: i64,
        kernel_size: [i64; 2],
        algo: KernelParamAlgo
    ) -> Self {
        let cache_size =
            cache_size::l1_cache_size().unwrap_or(32 * 1024) / std::mem::size_of::<T>();
        let (co_block_size, ci_block_size) = match algo {
            KernelParamAlgo::Heuristic => todo!(),
            KernelParamAlgo::Greedy => {
                find_exact_combination::<T, CONV_REGNUM>(
                    cache_size as i64,
                    out_channels as i64,
                    in_channels as i64,
                    kernel_size[1] as i64,
                    kernel_size[0] as i64
                )
            }
        };
        Self {
            l1_cache_size: cache_size::l1_cache_size().unwrap_or(32 * 1024) /
            std::mem::size_of::<T>(),
            out_channels,
            in_channels,
            kernel_size,
            _phantom: std::marker::PhantomData,
            co_block_size,
            ci_block_size,
        }
    }
    pub fn l1_cache_size(&self) -> usize {
        self.l1_cache_size * std::mem::size_of::<T>()
    }
    pub fn out_channels(&self) -> i64 {
        self.out_channels
    }
    pub fn in_channels(&self) -> i64 {
        self.in_channels
    }
    pub fn kernel_size(&self) -> [i64; 2] {
        self.kernel_size
    }
    pub fn set_out_channels(&mut self, out_channels: i64) -> &mut Self {
        self.out_channels = out_channels;
        self
    }
    pub fn set_in_channels(&mut self, in_channels: i64) -> &mut Self {
        self.in_channels = in_channels;
        self
    }
    pub fn set_kernel_size(&mut self, kernel_size: [i64; 2]) -> &mut Self {
        self.kernel_size = kernel_size;
        self
    }
}

fn find_exact_combination<T: CommonBounds, const REGNUM: usize>(
    max_cache_size: i64,
    max_co_b: i64,
    max_ci_b: i64,
    weight_size: i64,
    height_size: i64
) -> (i64, i64)
    where <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T> + VecSize
{
    let mut best_co_b = 1;
    let mut best_ci_b = 1;

    for ci_b in (1..max_ci_b + 1).rev() {
        for co_b in (1..max_co_b + 1)
            .rev()
            .filter(|&co_b| (co_b % (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64)) * 2 == 0) {
            let product =
                co_b * (REGNUM as i64) +
                weight_size * height_size * ci_b * ((REGNUM as i64) + co_b);

            if product <= max_cache_size && co_b <= max_co_b && ci_b <= max_ci_b {
                if ci_b > best_ci_b || (ci_b == best_ci_b && co_b > best_co_b) {
                    best_co_b = co_b;
                    best_ci_b = ci_b;
                }
            }
        }
    }
    let cache_line_size = (cache_size
        ::cache_line_size(1, cache_size::CacheType::Data)
        .unwrap_or(64) / std::mem::size_of::<T>()) as i64;
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

#[derive(Debug, Clone)]
pub struct Conv3dConfig<T> {
    pub(crate) l1_cache_size: usize,
    pub(crate) out_channels: i64,
    pub(crate) in_channels: i64,
    pub(crate) co_block_size: i64,
    pub(crate) ci_block_size: i64,
    pub(crate) kernel_size: [i64; 3],
    pub(crate) _phantom: std::marker::PhantomData<T>,
}

impl<T> Conv3dConfig<T> where T: CommonBounds {
    pub fn set_l1_cache_size(&mut self, l1_cache_size: usize) -> &mut Self {
        self.l1_cache_size = l1_cache_size / std::mem::size_of::<T>();
        self
    }
    pub fn set_co_block_size(&mut self, co_block_size: i64) -> &mut Self {
        self.co_block_size = co_block_size;
        self
    }
    pub fn set_ci_block_size(&mut self, ci_block_size: i64) -> &mut Self {
        self.ci_block_size = ci_block_size;
        self
    }
    pub fn new(
        out_channels: i64,
        in_channels: i64,
        kernel_size: [i64; 3],
        algo: KernelParamAlgo
    ) -> Self {
        let cache_size =
            cache_size::l1_cache_size().unwrap_or(32 * 1024) / std::mem::size_of::<T>();
        let (co_block_size, ci_block_size) = match algo {
            KernelParamAlgo::Heuristic => todo!(),
            KernelParamAlgo::Greedy => {
                find_exact_combination_3d::<T, CONV_REGNUM>(
                    cache_size as i64,
                    out_channels,
                    in_channels,
                    kernel_size[1],
                    kernel_size[0],
                    kernel_size[2]
                )
            }
        };
        Self {
            l1_cache_size: cache_size::l1_cache_size().unwrap_or(32 * 1024) /
            std::mem::size_of::<T>(),
            out_channels,
            in_channels,
            kernel_size,
            _phantom: std::marker::PhantomData,
            co_block_size,
            ci_block_size,
        }
    }
    pub fn l1_cache_size(&self) -> usize {
        self.l1_cache_size * std::mem::size_of::<T>()
    }
    pub fn out_channels(&self) -> i64 {
        self.out_channels
    }
    pub fn in_channels(&self) -> i64 {
        self.in_channels
    }
    pub fn kernel_size(&self) -> [i64; 3] {
        self.kernel_size
    }
    pub fn set_out_channels(&mut self, out_channels: i64) -> &mut Self {
        self.out_channels = out_channels;
        self
    }
    pub fn set_in_channels(&mut self, in_channels: i64) -> &mut Self {
        self.in_channels = in_channels;
        self
    }
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
    depth_size: i64
) -> (i64, i64)
    where <T as TypeCommon>::Vec: VecTrait<T> + Copy + Init<T> + VecSize
{
    let mut best_co_b = 0;
    let mut best_ci_b = 0;

    for co_b in (1..=max_co_b)
        .rev()
        .filter(|&co_b| co_b % (<<T as TypeCommon>::Vec as VecSize>::SIZE as i64) == 0) {
        for ci_b in (1..=max_ci_b).rev() {
            let product =
                co_b * (REGNUM as i64) +
                weight_size * height_size * depth_size * ci_b * ((REGNUM as i64) + co_b);

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
