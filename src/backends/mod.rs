use crate::{buffer::Buffer, dtype::DType};
use std::fmt::Debug;

pub mod cpu;
#[cfg(target_os = "macos")]
pub mod metal;

pub trait Device: Debug {
    type Buffer<Dtype: DType>: Buffer<Dtype>;
}
