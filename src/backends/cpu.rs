use super::Device;
use crate::{
    allocator::{Allocator, Buffer},
    dtype::DType,
};
use std::fmt::Debug;

pub struct CpuDevice;

impl Device for CpuDevice {
    type Allocator = CpuAllocator;
}

impl Debug for CpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CPU")
    }
}

#[derive(Debug, Clone)]
pub struct CpuBuffer<Dtype: DType> {
    raw_buffer: Box<[Dtype]>,
}

impl<Dtype: DType> Buffer<Dtype> for CpuBuffer<Dtype> {
    fn len(&self) -> usize {
        self.raw_buffer.len()
    }
    fn copy_in(&mut self, src: &[Dtype]) {
        assert_eq!(self.len(), src.len());
        self.raw_buffer.copy_from_slice(src);
    }

    fn copy_out(&self, dst: &mut [Dtype]) {
        assert_eq!(dst.len(), self.len());
        dst.copy_from_slice(&self.raw_buffer);
    }
}

pub struct CpuAllocator;

impl Allocator for CpuAllocator {
    type Buffer<Dtype: DType> = CpuBuffer<Dtype>;

    fn alloc<Dtype: DType>(size: usize) -> Result<Self::Buffer<Dtype>, std::alloc::AllocError> {
        // SAFETY: data has to be copied in the [`Bufffer`] before using it
        let raw_buffer: Box<[Dtype]> =
            unsafe { Box::<[Dtype]>::try_new_uninit_slice(size)?.assume_init() };
        Ok(CpuBuffer { raw_buffer })
    }

    unsafe fn free<Dtype: DType>(b: Self::Buffer<Dtype>) {
        drop(b); // explicit drop
    }
}

#[cfg(test)]
mod test {
    use crate::{allocator::Allocator, backends::cpu::CpuAllocator};
    use std::usize;

    #[test]
    fn oom() {
        assert!(CpuAllocator::alloc::<u8>(usize::MAX).is_err());
    }

    #[test]
    fn simple() {
        let _ = CpuAllocator::alloc::<f64>(16).unwrap();
    }
}
