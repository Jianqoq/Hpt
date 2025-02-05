use std::{fmt::Display, marker::PhantomData};

use hpt_traits::CommonBounds;

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
    #[allow(unused_mut)]
    #[allow(unused_assignments)]
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
            } else {
                panic!("Failed to get cache parameters");
            }
        }
        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            l1_line_size = std::str::from_utf8(
                &Command::new("sysctl")
                    .arg("hw.cachelinesize")
                    .output()
                    .expect("Failed to execute command")
                    .stdout,
            )
            .unwrap()
            .split(":")
            .last()
            .unwrap()
            .trim()
            .parse::<usize>()
            .unwrap_or(64);
            l1 = std::str::from_utf8(
                &Command::new("sysctl")
                    .arg("hw.l1dcachesize")
                    .output()
                    .expect("Failed to execute command")
                    .stdout,
            )
            .unwrap()
            .split(":")
            .last()
            .unwrap()
            .trim()
            .parse::<usize>()
            .unwrap();
            let l2_per_core = std::str::from_utf8(
                &Command::new("sysctl")
                    .arg("hw.perflevel0.cpusperl2")
                    .output()
                    .expect("Failed to execute command")
                    .stdout,
            )
            .unwrap()
            .split(":")
            .last()
            .unwrap()
            .trim()
            .parse::<usize>()
            .unwrap();
            l2 = std::str::from_utf8(
                &Command::new("sysctl")
                    .arg("hw.l2cachesize")
                    .output()
                    .expect("Failed to execute command")
                    .stdout,
            )
            .unwrap()
            .split(":")
            .last()
            .unwrap()
            .trim()
            .parse::<usize>()
            .unwrap()
                / l2_per_core;
            l1_line_size = std::str::from_utf8(
                &Command::new("sysctl")
                    .arg("hw.cachelinesize")
                    .output()
                    .expect("Failed to execute command")
                    .stdout,
            )
            .unwrap()
            .split(":")
            .last()
            .unwrap()
            .trim()
            .parse::<usize>()
            .unwrap();
            l2_line_size = l1_line_size;
            l3_line_size = l2_line_size;
            l1_associativity = 8;
            l2_associativity = 8;
            l3_associativity = 8;
            l1_sets = l1 / l1_associativity / l1_line_size;
            l2_sets = l2 / l2_associativity / l2_line_size;
            l3_sets = l3 / l3_associativity / l3_line_size;
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
