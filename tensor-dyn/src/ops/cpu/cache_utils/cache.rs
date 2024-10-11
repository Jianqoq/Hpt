use std::{fmt::Display, marker::PhantomData};

use tensor_traits::CommonBounds;

#[derive(Debug)]
pub(crate) struct Cache<T: CommonBounds> {
    pub(crate) l1: usize,
    pub(crate) l2: usize,
    pub(crate) l3: usize,
    pub(crate) l1_line_size: usize,
    pub(crate) l2_line_size: usize,
    pub(crate) l3_line_size: usize,
    pub(crate) l1_associativity: usize,
    pub(crate) l2_associativity: usize,
    pub(crate) l3_associativity: usize,
    pub(crate) l1_sets: usize,
    pub(crate) l2_sets: usize,
    pub(crate) l3_sets: usize,
    _marker: PhantomData<T>,
}

impl<T: CommonBounds> Cache<T> {
    pub(crate) fn new() -> Self {
        let mut l1 = 0;
        let mut l2 = 0;
        let mut l3 = 0;
        let mut l1_line_size = 0;
        let mut l2_line_size = 0;
        let mut l3_line_size = 0;
        let mut l1_associativity = 0;
        let mut l2_associativity = 0;
        let mut l3_associativity = 0;
        let mut l1_sets = 0;
        let mut l2_sets = 0;
        let mut l3_sets = 0;
        #[cfg(target_arch = "x86_64")]
        {
            use raw_cpuid::CpuId;
            let cpuid = CpuId::new();
            if let Some(cparams) = cpuid.get_cache_parameters() {
                for cache in cparams {
                    let size = cache.associativity()
                        * cache.physical_line_partitions()
                        * cache.coherency_line_size()
                        * cache.sets();
                    if cache.level() == 1 {
                        l1 = size;
                        l1_line_size = cache.coherency_line_size();
                        l1_associativity = cache.associativity();
                        l1_sets = cache.sets();
                    } else if cache.level() == 2 {
                        l2 = size;
                        l2_line_size = cache.coherency_line_size();
                        l2_associativity = cache.associativity();
                        l2_sets = cache.sets();
                    } else if cache.level() == 3 {
                        l3 = size;
                        l3_line_size = cache.coherency_line_size();
                        l3_associativity = cache.associativity();
                        l3_sets = cache.sets();
                    }
                }
            }
        }
        Self {
            l1: l1 / T::BIT_SIZE,
            l2: l2 / T::BIT_SIZE,
            l3: l3 / T::BIT_SIZE,
            l1_line_size: l1_line_size / T::BIT_SIZE,
            l2_line_size: l2_line_size / T::BIT_SIZE,
            l3_line_size: l3_line_size / T::BIT_SIZE,
            l1_associativity,
            l2_associativity,
            l3_associativity,
            l1_sets,
            l2_sets,
            l3_sets,
            _marker: PhantomData,
        }
    }
}

impl<T: CommonBounds> Display for Cache<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cache")
            .field("L1", &self.l1)
            .field("L2", &self.l2)
            .field("L3", &self.l3)
            .field("L1_line_size", &self.l1_line_size)
            .field("L2_line_size", &self.l2_line_size)
            .field("L3_line_size", &self.l3_line_size)
            .field("L1_associativity", &self.l1_associativity)
            .field("L2_associativity", &self.l2_associativity)
            .field("L3_associativity", &self.l3_associativity)
            .field("L1_sets", &self.l1_sets)
            .field("L2_sets", &self.l2_sets)
            .field("L3_sets", &self.l3_sets)
            .finish()
    }
}
