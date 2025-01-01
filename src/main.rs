use std::fs::File;

use struct_save::{
    load::{Load, MetaLoad},
    save::Save,
};
use tensor_dyn::*;
#[derive(Save, Load)]
struct SubModel {
    pub tensor: Tensor<f32>,
}

#[derive(Save, Load)]
struct Model {
    pub tensor: Tensor<f32>,
    pub sub_model: SubModel,
    pub eps: f32,
}

fn main() -> anyhow::Result<()> {
    let model = Model {
        tensor: Tensor::new([1.0, 2.0, 3.0, 4.0, 5.0]),
        sub_model: SubModel {
            tensor: Tensor::new([1.0, 2.0, 3.0]),
        },
        eps: 1e-6,
    };
    model.save("model.ftz")?;
    let model2 = Model::load("model.ftz")?;
    println!("model2: {:?}", model2.sub_model.tensor);
    Ok(())
}
