use crate::{buffer::Buffer, dtype::DType};
use std::fmt::Debug;

mod cpu;
pub use cpu::{CpuBuffer, CpuDevice};
// use crate::codegen::ir::Kernel;

#[cfg(target_os = "macos")]
mod metal;
#[cfg(target_os = "macos")]
pub use metal::{MetalBuffer, MetalDevice};

pub trait Device: 'static + Debug + Clone + Copy {
    type Buffer<Dtype: DType>: Buffer<Dtype>;
}

// pub trait Renderer {
//     fn render(kernel: &Kernel) -> String;
// }
