
pub(crate) const M_1_PI: f64 = 0.318309886183790671537767526745028724;
// Floating point limits
pub(crate) const SLEEF_FLT_MIN: f32 = 1.0e-126_f32; // 0x1p-126

pub(crate) const LOG10_2: f64 = 3.3219280948873623478703194294893901758648313930;

// Single precision base-10 logarithm constants
pub(crate) const L10_UF: f32 = 0.3010253906;
pub(crate) const L10_LF: f32 = 4.605038981e-6;
pub(crate) const L10_U: f64 = 0.30102999566383914498;
pub(crate) const L10_L: f64 = 1.4205023227266099418e-13;

// Single precision PI constants (alternative version)
pub(crate) const PI_A2F: f32 = 3.1414794921875;
pub(crate) const PI_B2F: f32 = 0.00011315941810607910156;
pub(crate) const PI_C2F: f32 = 1.9841872589410058936e-09;
pub(crate) const TRIGRANGEMAX2F: f32 = 125.0;

pub(crate) const PI_A2: f64 = 3.141592653589793116;
pub(crate) const PI_B2: f64 = 1.2246467991473532072e-16;
pub(crate) const TRIGRANGEMAX2: f64 = 15.0;

pub(crate) const PI_A: f64 = 3.1415926218032836914;
pub(crate) const PI_B: f64 = 3.1786509424591713469e-08;
pub(crate) const PI_C: f64 = 1.2246467864107188502e-16;
pub(crate) const PI_D: f64 = 1.2736634327021899816e-24;
pub(crate) const TRIGRANGEMAX: f64 = 15.0;

pub(crate) const M_2_PI_H: f64 = 0.63661977236758138243;
pub(crate) const M_2_PI_L: f64 = -3.9357353350364971764e-17;

pub(crate) const M_PI: f64 = 3.141592653589793238462643383279502884;

pub(crate) const R_LN2: f64 = 1.442695040888963407359924681001892137426645954152985934135449406931;
pub(crate) const L2U: f64 = 0.69314718055966295651160180568695068359375;
pub(crate) const L2L: f64 = 0.28235290563031577122588448175013436025525412068e-12;

pub(crate) const LOG1P_BOUND: f64 = 1.0e+307;

pub(crate) const LOG_DBL_MAX: f64 = 709.782712893384;

pub(crate) const SQRT_DBL_MAX: f64 = 1.3407807929942596355e+154;

pub(crate) const SLEEF_DBL_MIN: f64 = 2.2250738585072014e-308;

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

