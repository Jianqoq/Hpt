use std::{fs::File, marker::PhantomData};

use tensor_dyn::*;

pub struct ResNet {}

#[derive(Save, Load)]
pub struct Relu<T> {
    phantom: PhantomData<T>,
}

fn main() -> anyhow::Result<()> {
    Ok(())
}
