use crate::{buffer::Buffer, dtype::DType};
use std::fmt::Debug;

mod cpu;
pub use cpu::CpuDevice;

#[cfg(target_os = "macos")]
mod metal;
pub use metal::MetalDevice;

pub trait Device: Debug {
    type Buffer<Dtype: DType>: Buffer<Dtype>;
}
