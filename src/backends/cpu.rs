use crate::{
    allocator::{Allocator, Buffer},
    dtype::DType,
};

use super::Device;

pub struct CpuDevice;

impl Device for CpuDevice {
    type Allocator = CpuAllocator;

    fn allocator(&self) -> Self::Allocator {
        CpuAllocator
    }
}

impl<Dtype: DType> Buffer<Dtype> for Box<[Dtype]> {
    fn copy_in(&mut self, src: &[Dtype]) {
        assert_eq!(self.len(), src.len());
        self.copy_from_slice(src);
    }

    fn copy_out(&self, dst: &mut [Dtype]) {
        assert_eq!(dst.len(), self.len());
        dst.copy_from_slice(self);
    }
}

pub struct CpuAllocator;

impl Allocator for CpuAllocator {
    type Buffer<Dtype: DType> = Box<[Dtype]>;

    fn alloc<Dtype: crate::dtype::DType>(
        &self,
        size: usize,
    ) -> Result<Self::Buffer<Dtype>, std::alloc::AllocError> {
        // SAFETY: data has to be copied in the [`Bufffer`] before using it
        let array: Box<[Dtype]> =
            unsafe { Box::<[Dtype]>::try_new_uninit_slice(size)?.assume_init() };
        Ok(array)
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
