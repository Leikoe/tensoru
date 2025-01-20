use crate::{
    allocator::{Allocator, Buffer},
    dtype::DType,
};

use super::{Device, HasAllocator};

pub struct CpuDevice;

impl HasAllocator for CpuDevice {
    type Allocator<'device> = CpuAllocator;
}

impl Device for CpuDevice {
    fn allocator(&self) -> Self::Allocator<'static> {
        CpuAllocator
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

impl<'device> Allocator<'device> for CpuAllocator {
    type Buffer<Dtype: DType> = CpuBuffer<Dtype>;

    fn alloc<Dtype: crate::dtype::DType>(
        &self,
        size: usize,
    ) -> Result<Self::Buffer<Dtype>, std::alloc::AllocError> {
        // SAFETY: data has to be copied in the [`Bufffer`] before using it
        let raw_buffer: Box<[Dtype]> =
            unsafe { Box::<[Dtype]>::try_new_uninit_slice(size)?.assume_init() };
        Ok(CpuBuffer { raw_buffer })
    }

    unsafe fn free<Dtype: crate::dtype::DType>(&self, b: Self::Buffer<Dtype>) {
        drop(b); // explicit drop
    }
}

#[cfg(test)]
mod test {
    use std::usize;

    use crate::{
        allocator::Allocator,
        backends::{cpu::CpuDevice, Device},
    };

    #[test]
    fn oom() {
        let allocator = CpuDevice.allocator();
        assert!(allocator.alloc::<u8>(usize::MAX).is_err());
    }

    #[test]
    fn simple() {
        let allocator = CpuDevice.allocator();
        let _ = allocator.alloc::<f64>(16).unwrap();
    }
}
