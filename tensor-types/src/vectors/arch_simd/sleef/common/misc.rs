
pub(crate) const M_1_PI: f64 = 0.318309886183790671537767526745028724;
// Floating point limits
pub(crate) const SLEEF_FLT_MIN: f32 = 1.0e-126_f32; // 0x1p-126

pub(crate) const LOG10_2: f64 = 3.3219280948873623478703194294893901758648313930;

// Single precision base-10 logarithm constants
pub(crate) const L10_UF: f32 = 0.3010253906;
pub(crate) const L10_LF: f32 = 4.605038981e-6;

// Single precision PI constants (alternative version)
pub(crate) const PI_A2F: f32 = 3.1414794921875;
pub(crate) const PI_B2F: f32 = 0.00011315941810607910156;
pub(crate) const PI_C2F: f32 = 1.9841872589410058936e-09;
pub(crate) const TRIGRANGEMAX2F: f32 = 125.0;

// Maximum square root value for single precision
pub(crate) const SQRT_FLT_MAX: f64 = 18446743523953729536.0;

// Natural logarithm constants (single precision)
pub(crate) const L2_UF: f32 = 0.693145751953125;
pub(crate) const L2_LF: f32 = 1.428606765330187045e-06;

// Reciprocal of natural logarithm (single precision)
pub(crate) const R_LN2_F: f32 = 1.442695040888963407359924681001892137426645954152985934135449406931;

// Other bounds
/// log1p(f)(x) approximation holds up to these values
pub(crate) const LOG1PF_BOUND: f32 = 1.0e+38;  // 0x1.2ced32p+126