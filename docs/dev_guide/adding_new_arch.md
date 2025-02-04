# Adding New Arch Support

# Reason
If you found that the CPU you are using has the Simd instruction Hpt is not supported and you are willing to add support for that instruction like `avx512f`.

# How
1. All the Tensor operations are using scalar/simd computation under the hood. The scalar/simd computation is defined in tensor-types crate.

2. The folder `vectors` defines the simd vectors for simd registers with different size of bits. If your simd register is using `128-bit` and the instruction is not supported. You will want to go to `arch_simd/_128bit` folder, for each of the different type, you will need to add the computation for the new instruction.

Example:
```rust
impl PartialEq for f32x4 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let cmp = _mm_cmpeq_ps(self.0, other.0);
            _mm_movemask_ps(cmp) == -1
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let cmp = vceqq_f32(self.0, other.0);
            vminvq_u32(cmp) == 0xffffffff_u32
        }
        #[cfg(target_arch = "...")] // new arch
        unsafe {
            // new computation
        }
    }
}
```

3. The folder `scalar` defines the scalar computations. Most of the time you don't need to do anything on this folder unless for some scalar computation, your CPU has special instruction to perform that computation.

4. The file `convertion.rs` defines the type conversion traits and uses proc_macro defined from `tensor-macros` crate to auto implement the type conversion. If your CPU has special conversion instruction, you may want to go to the `tensor-macros` and modify the macro code.

5. The file `into_scalar` 