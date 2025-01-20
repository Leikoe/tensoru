use std::ops::{Add, Div, Mul, Sub};

pub trait DType:
    Sized
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + Copy
    + Clone
{
    fn zero() -> Self;
}

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
