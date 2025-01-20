use crate::dtype::DType;
use std::alloc::AllocError;

pub unsafe trait Allocator {
    type Buffer<Dtype: DType>: Buffer<Dtype>;
    fn alloc<Dtype: DType>(&self, size: usize) -> Result<Self::Buffer<Dtype>, AllocError>;
    fn free<Dtype: DType>(&self, b: Self::Buffer<Dtype>);
}

pub trait Buffer<Dtype: DType> {
    fn copy_in(&mut self, src: &[Dtype]);
    fn copy_out(&self, dst: &mut [Dtype]);
}
