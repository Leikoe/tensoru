use crate::backends::Device;
use crate::dtype::DType;
use std::alloc::AllocError;
use std::fmt::Debug;

pub trait Buffer<Dtype: DType>: Debug + Clone {
    fn new(size: usize) -> Result<Self, AllocError>;
    fn len(&self) -> usize;
    fn to<DEVICE: Device>(&self) -> DEVICE::Buffer<Dtype> {
        let mut tmp = vec![Dtype::ZERO; self.len()];
        self.copy_out(&mut tmp);
        let mut b = DEVICE::Buffer::<Dtype>::new(self.len()).unwrap();
        b.copy_in(&tmp);
        b
    }
    fn copy_in(&mut self, src: &[Dtype]);
    fn copy_out(&self, dst: &mut [Dtype]);
}
