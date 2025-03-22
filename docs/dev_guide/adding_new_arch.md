# Adding New Arch Support

### Things to know
If you found that the CPU you are using has the Simd instruction Hpt is not supported and you are willing to add support for that instruction like `avx512f`.

### How
1. All the Tensor operations are using implementation from `hpt-types/scalars/` and `hpt-types/vectors/arch_simd`.

2. The folder `vectors` defines the simd vectors for simd registers with different size of bits. If your simd register is using `128-bit` and the instruction is not supported. You will want to go to `arch_simd/_128bit` folder, create new folder for the new arch.

3. The folder `scalars` defines the scalar computations. Most of the time you don't need to do anything on this folder unless for some scalar computation, your CPU has special instruction to perform that computation.

4. The file `convertion.rs` defines the type conversion traits and uses proc_macro defined from `hpt-macros` crate to auto implement the type conversion. If your CPU has special conversion instruction, you may want to go to the `hpt-macros` and modify the macro code. The file `into_scalar.rs` is just using the method from `convertion.rs`, you won't need to worry about this file

5. After you finished all the steps above, you can start to run the hpt-tests crate and make sure all the tests passed.

6. Make a pull request.