use std::{
    fmt::Debug,
    ops::{Add, Div, Mul, Sub},
};

pub trait DType:
    Sized
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + Copy
    + Clone
    + Debug
    + 'static
{
    fn zero() -> Self;
}

// INTS
impl DType for u8 {
    fn zero() -> Self {
        0
    }
}
impl DType for u16 {
    fn zero() -> Self {
        0
    }
}
impl DType for u32 {
    fn zero() -> Self {
        0
    }
}
impl DType for u64 {
    fn zero() -> Self {
        0
    }
}

// FLOATS
impl DType for f16 {
    fn zero() -> Self {
        0.
    }
}
impl DType for f32 {
    fn zero() -> Self {
        0.
    }
}
impl DType for f64 {
    fn zero() -> Self {
        0.
    }
}
