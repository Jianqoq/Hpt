use crate::{ dtype::TypeCommon, traits::SimdMath };
use crate::vectors::traits::VecTrait;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// a trait to define constants for sleef
pub trait SleefConstants {
    /// the maximum value for the trigonometric range
    const TRIGRANGEMAX2f: f32;
}

fn vcast_vf_f(val: f32) -> <f32 as TypeCommon>::Vec {
    <f32 as TypeCommon>::Vec::splat(val)
}

fn vreinterpret_vm_vf(vec: <f32 as TypeCommon>::Vec) -> <f32 as TypeCommon>::Vec {
    #[cfg(target_arch = "x86_64")]
    return unsafe {
        std::mem::transmute(_mm256_castps_si256(std::mem::transmute(vec)))
    };
}

fn vabs_vf_vf(vec: <f32 as TypeCommon>::Vec) -> <f32 as TypeCommon>::Vec {
    let a = vreinterpret_vm_vf(vcast_vf_f(-0.0f32));

    todo!()
}

impl<T> SimdMath for T where T: Copy {
    fn sin(self) -> Self {
        let (u, s, t) = (self, self, self);

        todo!()
    }
}
