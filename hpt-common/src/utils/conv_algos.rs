/// enum for conv algorithms
pub enum ConvAlgo {
    /// ImplicitGemm
    ImplicitGemm,
    /// ImplicitPrecompGemm
    ImplicitPrecompGemm,
    /// Gemm
    Gemm,
    /// Direct
    Direct,
    /// Fft
    Fft,
    /// FftTiling
    FftTiling,
    /// Winograd
    Winograd,
    /// WinogradNonFused
    WinogradNonFused,
    /// Count
    Count,
}
