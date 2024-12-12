use std::arch::x86_64::*;

use crate::sleef_types::*;

#[inline(always)]
pub(crate) unsafe fn vtestallones_i_vo32(g: VMask) -> i32 {
    // 检查是否所有位都为1
    (_mm_movemask_epi8(g) == 0xFFFF) as i32
}

#[inline(always)]
pub(crate) unsafe fn vtestallones_i_vo64(g: VMask) -> i32 {
    // 检查是否所有位都为1 (64位版本)
    (_mm_movemask_epi8(g) == 0xFFFF) as i32
}

#[inline(always)]
pub(crate) unsafe fn vstoreu_v_p_vi(p: *mut i32, v: VInt) {
    // 存储128位整数到未对齐的内存
    _mm_storeu_si128(p as *mut __m128i, v);
}

// vmask 之间的操作
#[inline(always)]
pub(crate) unsafe fn vand_vm_vm_vm(x: VMask, y: VMask) -> VMask {
    _mm_and_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vm_vm_vm(x: VMask, y: VMask) -> VMask {
    _mm_andnot_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vor_vm_vm_vm(x: VMask, y: VMask) -> VMask {
    _mm_or_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vxor_vm_vm_vm(x: VMask, y: VMask) -> VMask {
    _mm_xor_si128(x, y)
}

// vopmask 之间的操作
#[inline(always)]
pub(crate) unsafe fn vand_vo_vo_vo(x: Vopmask, y: Vopmask) -> Vopmask {
    _mm_and_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vo_vo_vo(x: Vopmask, y: Vopmask) -> Vopmask {
    _mm_andnot_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vor_vo_vo_vo(x: Vopmask, y: Vopmask) -> Vopmask {
    _mm_or_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vxor_vo_vo_vo(x: Vopmask, y: Vopmask) -> Vopmask {
    _mm_xor_si128(x, y)
}

// 64位 vopmask 和 vmask 之间的操作
#[inline(always)]
pub(crate) unsafe fn vand_vm_vo64_vm(x: Vopmask, y: VMask) -> VMask {
    _mm_and_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vor_vm_vo64_vm(x: Vopmask, y: VMask) -> VMask {
    _mm_or_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vm_vo64_vm(x: VMask, y: VMask) -> VMask {
    _mm_andnot_si128(x, y)
}

// 32位 vopmask 和 vmask 之间的操作
#[inline(always)]
pub(crate) unsafe fn vand_vm_vo32_vm(x: Vopmask, y: VMask) -> VMask {
    _mm_and_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vor_vm_vo32_vm(x: Vopmask, y: VMask) -> VMask {
    _mm_or_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vm_vo32_vm(x: VMask, y: VMask) -> VMask {
    _mm_andnot_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vo32_vo64(m: Vopmask) -> Vopmask {
    // 将64位掩码转换为32位掩码
    // 0x08 = 0b00001000 指定了重排模式
    _mm_shuffle_epi32(m, 0x08)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vo64_vo32(m: Vopmask) -> Vopmask {
    // 将32位掩码转换为64位掩码
    // 0x50 = 0b01010000 指定了重排模式
    _mm_shuffle_epi32(m, 0x50)
}

#[inline(always)]
pub(crate) unsafe fn vrint_vi_vd(vd: VDouble) -> VInt {
    // 将双精度浮点数向量转换为整数向量（四舍五入）
    _mm_cvtpd_epi32(vd)
}

#[inline(always)]
pub(crate) unsafe fn vtruncate_vi_vd(vd: VDouble) -> VInt {
    // 将双精度浮点数向量转换为整数向量（截断）
    _mm_cvttpd_epi32(vd)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vd_vi(vi: VInt) -> VDouble {
    // 将整数向量转换为双精度浮点数向量
    _mm_cvtepi32_pd(vi)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vi_i(i: i32) -> VInt {
    // 创建一个整数向量，低两个元素为i，高两个元素为0
    _mm_set_epi32(0, 0, i, i)
}

#[inline(always)]
pub(crate) unsafe fn vcastu_vm_vi(vi: VInt) -> VInt2 {
    // 将整数向量转换为掩码向量，并进行位模式重排
    // 0x73 = 0b01110011 指定重排模式
    _mm_and_si128(_mm_shuffle_epi32(vi, 0x73), _mm_set_epi32(-1, 0, -1, 0))
}

#[inline(always)]
pub(crate) unsafe fn vcastu_vi_vm(vi: VInt2) -> VInt {
    // 将掩码向量转换回整数向量
    // 0x0d = 0b00001101 指定重排模式
    _mm_shuffle_epi32(vi, 0x0d)
}

#[inline(always)]
pub(crate) unsafe fn vtruncate_vd_vd(vd: VDouble) -> VDouble {
    // 双精度浮点数截断（去掉小数部分）
    // 先转为整数再转回浮点数
    vcast_vd_vi(vtruncate_vi_vd(vd))
}

#[inline(always)]
pub(crate) unsafe fn vrint_vd_vd(vd: VDouble) -> VDouble {
    // 双精度浮点数四舍五入
    // 先转为整数再转回浮点数
    vcast_vd_vi(vrint_vi_vd(vd))
}

#[inline(always)]
pub(crate) unsafe fn veq64_vo_vm_vm(x: VMask, y: VMask) -> Vopmask {
    // 比较两个64位掩码是否相等
    // 0xb1 = 0b10110001 用于重排32位元素
    let t = _mm_cmpeq_epi32(x, y);
    // 将相邻的32位比较结果进行与运算
    vand_vm_vm_vm(t, _mm_shuffle_epi32(t, 0xb1))
}

#[inline(always)]
pub(crate) unsafe fn vadd64_vm_vm_vm(x: VMask, y: VMask) -> VMask {
    // 64位整数向量加法
    _mm_add_epi64(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vm_i_i(i0: i32, i1: i32) -> VMask {
    // 创建一个掩码向量，交替放置i0和i1
    // 结果为 [i0, i1, i0, i1]
    _mm_set_epi32(i0, i1, i0, i1)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vd_d(d: f64) -> VDouble {
    // 创建所有元素都为d的向量
    _mm_set1_pd(d)
}

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vm_vd(vd: VDouble) -> VMask {
    // 将双精度向量重解释为掩码
    _mm_castpd_si128(vd)
}

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vd_vm(vm: VMask) -> VDouble {
    // 将掩码重解释为双精度向量
    _mm_castsi128_pd(vm)
}

// 基本算术运算
#[inline(always)]
pub(crate) unsafe fn vadd_vd_vd_vd(x: VDouble, y: VDouble) -> VDouble {
    _mm_add_pd(x, y) // 向量加法
}

#[inline(always)]
pub(crate) unsafe fn vsub_vd_vd_vd(x: VDouble, y: VDouble) -> VDouble {
    _mm_sub_pd(x, y) // 向量减法
}

#[inline(always)]
pub(crate) unsafe fn vmul_vd_vd_vd(x: VDouble, y: VDouble) -> VDouble {
    _mm_mul_pd(x, y) // 向量乘法
}

#[inline(always)]
pub(crate) unsafe fn vrec_vd_vd(x: VDouble) -> VDouble {
    // 向量倒数 (1/x)
    _mm_div_pd(_mm_set1_pd(1.0), x)
}

#[inline(always)]
pub(crate) unsafe fn vsqrt_vd_vd(x: VDouble) -> VDouble {
    _mm_sqrt_pd(x) // 向量平方根
}

#[inline(always)]
pub(crate) unsafe fn vabs_vd_vd(d: VDouble) -> VDouble {
    // 向量绝对值
    _mm_andnot_pd(_mm_set1_pd(-0.0), d)
}

#[inline(always)]
pub(crate) unsafe fn vneg_vd_vd(d: VDouble) -> VDouble {
    // 向量取反
    _mm_xor_pd(_mm_set1_pd(-0.0), d)
}

// 组合运算
#[inline(always)]
pub(crate) unsafe fn vmla_vd_vd_vd_vd(x: VDouble, y: VDouble, z: VDouble) -> VDouble {
    // 乘加: (x * y) + z
    vadd_vd_vd_vd(vmul_vd_vd_vd(x, y), z)
}

#[inline(always)]
pub(crate) unsafe fn vmlapn_vd_vd_vd_vd(x: VDouble, y: VDouble, z: VDouble) -> VDouble {
    // 乘加取反: (x * y) - z
    vsub_vd_vd_vd(vmul_vd_vd_vd(x, y), z)
}

#[inline(always)]
pub(crate) unsafe fn vmax_vd_vd_vd(x: VDouble, y: VDouble) -> VDouble {
    _mm_max_pd(x, y) // 向量最大值
}

#[inline(always)]
pub(crate) unsafe fn vmin_vd_vd_vd(x: VDouble, y: VDouble) -> VDouble {
    _mm_min_pd(x, y) // 向量最小值
}

#[inline(always)]
pub(crate) unsafe fn veq_vo_vd_vd(x: VDouble, y: VDouble) -> Vopmask {
    // 等于比较 (x == y)
    _mm_castpd_si128(_mm_cmpeq_pd(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vneq_vo_vd_vd(x: VDouble, y: VDouble) -> Vopmask {
    // 不等于比较 (x != y)
    _mm_castpd_si128(_mm_cmpneq_pd(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vlt_vo_vd_vd(x: VDouble, y: VDouble) -> Vopmask {
    // 小于比较 (x < y)
    _mm_castpd_si128(_mm_cmplt_pd(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vle_vo_vd_vd(x: VDouble, y: VDouble) -> Vopmask {
    // 小于等于比较 (x <= y)
    _mm_castpd_si128(_mm_cmple_pd(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vgt_vo_vd_vd(x: VDouble, y: VDouble) -> Vopmask {
    // 大于比较 (x > y)
    _mm_castpd_si128(_mm_cmpgt_pd(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vge_vo_vd_vd(x: VDouble, y: VDouble) -> Vopmask {
    // 大于等于比较 (x >= y)
    _mm_castpd_si128(_mm_cmpge_pd(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vadd_vi_vi_vi(x: VInt, y: VInt) -> VInt {
    // 32位整数向量加法
    _mm_add_epi32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vsub_vi_vi_vi(x: VInt, y: VInt) -> VInt {
    // 32位整数向量减法
    _mm_sub_epi32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vneg_vi_vi(e: VInt) -> VInt {
    // 32位整数向量取反
    // 通过 0 - e 实现
    vsub_vi_vi_vi(vcast_vi_i(0), e)
}

#[inline(always)]
pub(crate) unsafe fn vand_vi_vi_vi(x: VInt, y: VInt) -> VInt {
    // 整数向量按位与运算
    _mm_and_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vi_vi_vi(x: VInt, y: VInt) -> VInt {
    // 整数向量按位与非运算
    // 计算 (!x) & y
    _mm_andnot_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vor_vi_vi_vi(x: VInt, y: VInt) -> VInt {
    // 整数向量按位或运算
    _mm_or_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vand_vi_vo_vi(x: Vopmask, y: VInt) -> VInt {
    // 掩码和整数向量的按位与运算
    // 用于条件选择：保留掩码为1的位
    _mm_and_si128(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vsll_vi_vi_i<const C: i32>(x: VInt) -> VInt {
    // 逻辑左移（填充0）
    _mm_slli_epi32::<C>(x)
}

#[inline(always)]
pub(crate) unsafe fn vsrl_vi_vi_i<const C: i32>(x: VInt) -> VInt {
    // 逻辑右移（填充0）
    _mm_srli_epi32::<C>(x)
}

#[inline(always)]
pub(crate) unsafe fn vsra_vi_vi_i<const C: i32>(x: VInt) -> VInt {
    // 算术右移（填充符号位）
    _mm_srai_epi32::<C>(x)
}

#[inline(always)]
pub(crate) unsafe fn veq_vo_vi_vi(x: VInt, y: VInt) -> Vopmask {
    // 相等比较，返回操作掩码
    _mm_cmpeq_epi32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vgt_vo_vi_vi(x: VInt, y: VInt) -> Vopmask {
    // 大于比较，返回操作掩码
    _mm_cmpgt_epi32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vsel_vi_vo_vi_vi(m: Vopmask, x: VInt, y: VInt) -> VInt {
    // 根据掩码m选择x或y的对应位
    // 对每一位：m ? x : y
    vor_vm_vm_vm(
        vand_vm_vm_vm(m, x),    // m & x
        vandnot_vm_vm_vm(m, y), // (!m) & y
    )
}

#[inline(always)]
pub(crate) unsafe fn vsel_vd_vo_vd_vd(opmask: Vopmask, x: VDouble, y: VDouble) -> VDouble {
    // 根据掩码选择双精度浮点数向量的值
    // 对每一位：opmask ? x : y
    _mm_or_pd(
        _mm_and_pd(_mm_castsi128_pd(opmask), x),    // mask & x
        _mm_andnot_pd(_mm_castsi128_pd(opmask), y), // (!mask) & y
    )
}

#[inline(always)]
pub(crate) unsafe fn vsel_vd_vo_d_d(o: Vopmask, v1: f64, v0: f64) -> VDouble {
    // 根据掩码选择两个标量值
    vsel_vd_vo_vd_vd(o, vcast_vd_d(v1), vcast_vd_d(v0))
}

#[inline(always)]
pub(crate) unsafe fn visinf_vo_vd(d: VDouble) -> Vopmask {
    // 检测是否为无穷大（正或负）
    vreinterpret_vm_vd(_mm_cmpeq_pd(vabs_vd_vd(d), _mm_set1_pd(f64::INFINITY)))
}

#[inline(always)]
pub(crate) unsafe fn vispinf_vo_vd(d: VDouble) -> Vopmask {
    // 检测是否为正无穷大
    vreinterpret_vm_vd(_mm_cmpeq_pd(d, _mm_set1_pd(f64::INFINITY)))
}

#[inline(always)]
pub(crate) unsafe fn visnan_vo_vd(d: VDouble) -> Vopmask {
    // 检测是否为NaN（不是数字）
    // NaN的特性：它不等于它自己
    vreinterpret_vm_vd(_mm_cmpneq_pd(d, d))
}

#[inline(always)]
pub(crate) unsafe fn vgather_vd_p_vi(ptr: *const f64, vi: VInt) -> VDouble {
    // 根据索引向量从内存收集双精度值
    let mut a = [0i32; std::mem::size_of::<VInt>() / std::mem::size_of::<i32>()];
    vstoreu_v_p_vi(a.as_mut_ptr(), vi);
    _mm_set_pd(*ptr.add(a[1] as usize), *ptr.add(a[0] as usize))
}

#[inline(always)]
pub(crate) unsafe fn vcast_vm_vi2(vi: VInt2) -> VMask {
    // 128位整数向量到掩码的转换
    vi
}

#[inline(always)]
pub(crate) unsafe fn vrint_vi2_vf(vf: VFloat) -> VInt2 {
    // 单精度浮点向量到整数向量的转换（四舍五入）
    _mm_cvtps_epi32(vf)
}

#[inline(always)]
pub(crate) unsafe fn vtruncate_vi2_vf(vf: VFloat) -> VInt2 {
    // 单精度浮点向量到整数向量的转换（截断）
    _mm_cvttps_epi32(vf)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vf_vi2(vi: VInt2) -> VFloat {
    // 整数向量到单精度浮点向量的转换
    _mm_cvtepi32_ps(vcast_vm_vi2(vi))
}

#[inline(always)]
pub(crate) unsafe fn vcast_vf_f(f: f32) -> VFloat {
    // 标量浮点数到向量的广播
    _mm_set1_ps(f)
}

#[inline(always)]
pub(crate) unsafe fn vcast_vi2_i(i: i32) -> VInt2 {
    // 标量整数到向量的广播
    _mm_set1_epi32(i)
}

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vm_vf(vf: VFloat) -> VMask {
    // 浮点向量到掩码的位模式重解释
    _mm_castps_si128(vf)
}

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vf_vm(vm: VMask) -> VFloat {
    // 掩码到浮点向量的位模式重解释
    _mm_castsi128_ps(vm)
}

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vf_vi2(vm: VInt2) -> VFloat {
    // 整数向量到浮点向量的位模式重解释
    _mm_castsi128_ps(vm)
}

#[inline(always)]
pub(crate) unsafe fn vreinterpret_vi2_vf(vf: VFloat) -> VInt2 {
    // 浮点向量到整数向量的位模式重解释
    _mm_castps_si128(vf)
}

#[inline(always)]
pub(crate) unsafe fn vtruncate_vf_vf(vd: VFloat) -> VFloat {
    // 向零截断
    vcast_vf_vi2(vtruncate_vi2_vf(vd))
}

#[inline(always)]
pub(crate) unsafe fn vrint_vf_vf(vf: VFloat) -> VFloat {
    // 四舍五入到最近整数
    vcast_vf_vi2(vrint_vi2_vf(vf))
}

// 基本算术运算
#[inline(always)]
pub(crate) unsafe fn vadd_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    _mm_add_ps(x, y) // 向量加法
}

#[inline(always)]
pub(crate) unsafe fn vsub_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    _mm_sub_ps(x, y) // 向量减法
}

#[inline(always)]
pub(crate) unsafe fn vmul_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    _mm_mul_ps(x, y) // 向量乘法
}

#[inline(always)]
pub(crate) unsafe fn vdiv_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    _mm_div_ps(x, y) // 向量除法
}

#[inline(always)]
pub(crate) unsafe fn vrec_vf_vf(x: VFloat) -> VFloat {
    // 向量倒数 (1/x)
    vdiv_vf_vf_vf(vcast_vf_f(1.0), x)
}

#[inline(always)]
pub(crate) unsafe fn vsqrt_vf_vf(x: VFloat) -> VFloat {
    _mm_sqrt_ps(x) // 向量平方根
}

#[inline(always)]
pub(crate) unsafe fn vabs_vf_vf(f: VFloat) -> VFloat {
    // 向量绝对值
    vreinterpret_vf_vm(vandnot_vm_vm_vm(
        vreinterpret_vm_vf(vcast_vf_f(-0.0)),
        vreinterpret_vm_vf(f),
    ))
}

#[inline(always)]
pub(crate) unsafe fn vneg_vf_vf(d: VFloat) -> VFloat {
    // 向量取反
    vreinterpret_vf_vm(vxor_vm_vm_vm(
        vreinterpret_vm_vf(vcast_vf_f(-0.0)),
        vreinterpret_vm_vf(d),
    ))
}

// 复合运算
#[inline(always)]
pub(crate) unsafe fn vmla_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
    // 乘加: (x * y) + z
    vadd_vf_vf_vf(vmul_vf_vf_vf(x, y), z)
}

#[inline(always)]
pub(crate) unsafe fn vmlanp_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
    // 乘加取反2: z - (x * y)
    vsub_vf_vf_vf(z, vmul_vf_vf_vf(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vmax_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    _mm_max_ps(x, y) // 向量最大值
}

#[inline(always)]
pub(crate) unsafe fn vmin_vf_vf_vf(x: VFloat, y: VFloat) -> VFloat {
    _mm_min_ps(x, y) // 向量最小值
}

// 单精度浮点数向量比较操作
#[inline(always)]
pub(crate) unsafe fn veq_vo_vf_vf(x: VFloat, y: VFloat) -> Vopmask {
    // 等于比较 (x == y)
    vreinterpret_vm_vf(_mm_cmpeq_ps(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vneq_vo_vf_vf(x: VFloat, y: VFloat) -> Vopmask {
    // 不等于比较 (x != y)
    vreinterpret_vm_vf(_mm_cmpneq_ps(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vlt_vo_vf_vf(x: VFloat, y: VFloat) -> Vopmask {
    // 小于比较 (x < y)
    vreinterpret_vm_vf(_mm_cmplt_ps(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vle_vo_vf_vf(x: VFloat, y: VFloat) -> Vopmask {
    // 小于等于比较 (x <= y)
    vreinterpret_vm_vf(_mm_cmple_ps(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vgt_vo_vf_vf(x: VFloat, y: VFloat) -> Vopmask {
    // 大于比较 (x > y)
    vreinterpret_vm_vf(_mm_cmpgt_ps(x, y))
}

#[inline(always)]
pub(crate) unsafe fn vge_vo_vf_vf(x: VFloat, y: VFloat) -> Vopmask {
    // 大于等于比较 (x >= y)
    vreinterpret_vm_vf(_mm_cmpge_ps(x, y))
}

// 128位整数向量运算
#[inline(always)]
pub(crate) unsafe fn vadd_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
    // 整数向量加法
    vadd_vi_vi_vi(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vsub_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
    // 整数向量减法
    vsub_vi_vi_vi(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vneg_vi2_vi2(e: VInt2) -> VInt2 {
    // 整数向量取反
    vsub_vi2_vi2_vi2(vcast_vi2_i(0), e)
}

// 128位整数向量位运算
#[inline(always)]
pub(crate) unsafe fn vand_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
    // 按位与
    vand_vi_vi_vi(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vandnot_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
    // 按位与非
    vandnot_vi_vi_vi(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vor_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
    // 按位或
    vor_vi_vi_vi(x, y)
}

// 掩码和整数向量的位运算
#[inline(always)]
pub(crate) unsafe fn vand_vi2_vo_vi2(x: Vopmask, y: VInt2) -> VInt2 {
    // 掩码和整数向量的按位与
    vand_vi_vo_vi(x, y)
}

// 整数向量移位操作
#[inline(always)]
pub(crate) unsafe fn vsll_vi2_vi2_i<const C: i32>(x: VInt2) -> VInt2 {
    // 逻辑左移
    vsll_vi_vi_i::<C>(x)
}

#[inline(always)]
pub(crate) unsafe fn vsrl_vi2_vi2_i<const C: i32>(x: VInt2) -> VInt2 {
    // 逻辑右移（填充0）
    vsrl_vi_vi_i::<C>(x)
}

#[inline(always)]
pub(crate) unsafe fn vsra_vi2_vi2_i<const C: i32>(x: VInt2) -> VInt2 {
    // 算术右移（填充符号位）
    vsra_vi_vi_i::<C>(x)
}

// 整数向量比较操作
#[inline(always)]
pub(crate) unsafe fn veq_vo_vi2_vi2(x: VInt2, y: VInt2) -> Vopmask {
    // 等于比较，返回掩码
    _mm_cmpeq_epi32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vgt_vo_vi2_vi2(x: VInt2, y: VInt2) -> Vopmask {
    // 大于比较，返回掩码
    _mm_cmpgt_epi32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vgt_vi2_vi2_vi2(x: VInt2, y: VInt2) -> VInt2 {
    // 大于比较，返回整数向量
    _mm_cmpgt_epi32(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vsel_vi2_vo_vi2_vi2(m: Vopmask, x: VInt2, y: VInt2) -> VInt2 {
    // 根据掩码选择整数向量的值
    vor_vi2_vi2_vi2(
        vand_vi2_vi2_vi2(m, x),    // m & x
        vandnot_vi2_vi2_vi2(m, y), // (!m) & y
    )
}

#[inline(always)]
pub(crate) unsafe fn vsel_vf_vo_vf_vf(opmask: Vopmask, x: VFloat, y: VFloat) -> VFloat {
    // 根据掩码选择浮点向量的值
    _mm_or_ps(
        _mm_and_ps(_mm_castsi128_ps(opmask), x),
        _mm_andnot_ps(_mm_castsi128_ps(opmask), y),
    )
}

#[inline(always)]
pub(crate) unsafe fn vsel_vf_vo_f_f(o: Vopmask, v1: f32, v0: f32) -> VFloat {
    // 根据掩码选择两个标量值
    vsel_vf_vo_vf_vf(o, vcast_vf_f(v1), vcast_vf_f(v0))
}

// 特殊值检测
#[inline(always)]
pub(crate) unsafe fn visinf_vo_vf(d: VFloat) -> Vopmask {
    // 检测是否为无穷大
    veq_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(f32::INFINITY))
}

#[inline(always)]
pub(crate) unsafe fn vispinf_vo_vf(d: VFloat) -> Vopmask {
    // 检测是否为正无穷大
    veq_vo_vf_vf(d, vcast_vf_f(f32::INFINITY))
}

#[inline(always)]
pub(crate) unsafe fn visnan_vo_vf(d: VFloat) -> Vopmask {
    // 检测是否为NaN
    vneq_vo_vf_vf(d, d)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vfma_vd_vd_vd_vd(x: VDouble, y: VDouble, z: VDouble) -> VDouble {
    _mm_fmadd_pd(x, y, z)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vfmanp_vd_vd_vd_vd(x: VDouble, y: VDouble, z: VDouble) -> VDouble {
    _mm_fnmadd_pd(x, y, z)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vfmapn_vd_vd_vd_vd(x: VDouble, y: VDouble, z: VDouble) -> VDouble {
    _mm_fmsub_pd(x, y, z)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vfma_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
    _mm_fmadd_ps(x, y, z)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vfmanp_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
    _mm_fnmadd_ps(x, y, z)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub(crate) unsafe fn vfmapn_vf_vf_vf_vf(x: VFloat, y: VFloat, z: VFloat) -> VFloat {
    _mm_fmsub_ps(x, y, z)
}

#[inline(always)]
pub(crate) unsafe fn vsrl64_vm_vm_i<const C: i32>(x: VMask) -> VMask {
    _mm_srli_epi64::<C>(x)
}

#[inline(always)]
pub(crate) unsafe fn vsub64_vm_vm_vm(x: VMask, y: VMask) -> VMask {
    _mm_sub_epi64(x, y)
}

#[inline(always)]
pub(crate) unsafe fn vgather_vf_p_vi2(ptr: *const f32, vi2: VInt2) -> VFloat {
    _mm_i32gather_ps(ptr, vi2, 4)
}
