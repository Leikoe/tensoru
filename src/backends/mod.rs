use crate::{buffer::Buffer, dtype::DType};
use std::fmt::Debug;

mod cpu;
use crate::codegen::ir::Kernel;
pub use cpu::{CpuBuffer, CpuDevice};

#[cfg(target_os = "macos")]
mod metal;
#[cfg(target_os = "macos")]
pub use metal::{MetalBuffer, MetalDevice};

pub trait Device: 'static + Debug + Clone + Copy {
    type Buffer<Dtype: DType>: Buffer<Dtype>;
}

pub trait Render {
    fn render(kernel: &Kernel) -> String;
}
