use crate::{buffer::Buffer, dtype::DType};
use std::fmt::Debug;

mod cpu;
pub use cpu::{CpuBuffer, CpuDevice};

#[cfg(target_os = "macos")]
mod metal;
pub use metal::{MetalBuffer, MetalDevice};

pub trait Device: Debug {
    type Buffer<Dtype: DType>: Buffer<Dtype>;
}

pub trait Renderer {
    // fn
}
