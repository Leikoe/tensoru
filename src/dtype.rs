use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

pub trait DType:
    Sized
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + AddAssign<Self>
    + SubAssign<Self>
    + MulAssign<Self>
    + DivAssign<Self>
    + Copy
    + Clone
    + Debug
    + 'static
{
    const ZERO: Self;
}

// INTS
impl DType for u8 {
    const ZERO: Self = 0;
}
impl DType for u16 {
    const ZERO: Self = 0;
}
impl DType for u32 {
    const ZERO: Self = 0;
}
impl DType for u64 {
    const ZERO: Self = 0;
}

// FLOATS
impl DType for f16 {
    const ZERO: Self = 0.;
}
impl DType for f32 {
    const ZERO: Self = 0.;
}
impl DType for f64 {
    const ZERO: Self = 0.;
}
