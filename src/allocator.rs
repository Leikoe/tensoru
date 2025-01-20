use crate::dtype::DType;
use std::alloc::AllocError;
use std::fmt::Debug;

pub trait Allocator {
    type Buffer<Dtype: DType>: Buffer<Dtype>;
    fn alloc<Dtype: DType>(size: usize) -> Result<Self::Buffer<Dtype>, AllocError>;
    unsafe fn free<Dtype: DType>(b: Self::Buffer<Dtype>);
}

pub trait Buffer<Dtype: DType>: Debug + Clone {
    fn len(&self) -> usize;
    fn copy_in(&mut self, src: &[Dtype]);
    fn copy_out(&self, dst: &mut [Dtype]);
}
