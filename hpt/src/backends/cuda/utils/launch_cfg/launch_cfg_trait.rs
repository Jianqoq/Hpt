use cudarc::driver::LaunchConfig;

pub(crate) trait LaunchConfigUtils {
    #[allow(unused)]
    fn block_size(&self) -> u32;
    #[allow(unused)]
    fn grid_size(&self) -> u32;
}

impl LaunchConfigUtils for LaunchConfig {
    fn block_size(&self) -> u32 {
        self.block_dim.0 * self.block_dim.1 * self.block_dim.2
    }

    fn grid_size(&self) -> u32 {
        self.grid_dim.0 * self.grid_dim.1 * self.grid_dim.2
    }
}
