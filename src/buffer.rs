use crate::dtype::DType;
use std::alloc::AllocError;
use std::fmt::Debug;

pub trait Buffer<Dtype: DType>: Debug + Clone + Drop {
    fn new(size: usize) -> Result<Self, AllocError>;
    fn len(&self) -> usize;
    fn copy_in(&mut self, src: &[Dtype]);
    fn copy_out(&self, dst: &mut [Dtype]);
}
