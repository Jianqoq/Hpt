use tensor_common::pointer::Pointer;
use tensor_types::dtype::TypeCommon;

pub(crate) trait ConvKernel: Sized + TypeCommon {
    fn conv2d_nhwc(
        inp_offset: i64,
        kernel_offset: i64,
        isw_sw: i64,
        inp: &Pointer<Self>,
        kernel: &Pointer<Self>,
        results: &mut [<Self as TypeCommon>::Vec],
    );
    fn conv2d_nhwc_store(
        out_offset: i64,
        osw: i64,
        results: &[<Self as TypeCommon>::Vec],
        out: &mut Pointer<Self>,
    );
}
