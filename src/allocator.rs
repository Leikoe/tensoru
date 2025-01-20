use crate::backends::cpu::CpuDevice;
use crate::backends::Device;
use crate::dtype::DType;
use std::alloc::AllocError;
use std::fmt::Debug;

pub trait Allocator<'device> {
    type Buffer<Dtype: DType>: Buffer<Dtype>;
    fn alloc<Dtype: DType>(&self, size: usize) -> Result<Self::Buffer<Dtype>, AllocError>;
    unsafe fn free<Dtype: DType>(&self, b: Self::Buffer<Dtype>);
}

pub trait Buffer<Dtype: DType>: Debug + Clone {
    fn len(&self) -> usize;
    fn copy_in(&mut self, src: &[Dtype]);
    fn copy_out(&self, dst: &mut [Dtype]);
}
