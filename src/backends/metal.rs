use metal::{
    foreign_types::ForeignType, Buffer as RawBuffer, Device as RawDevice, MTLResourceOptions,
};
use std::{alloc::AllocError, sync::Arc};

use crate::{
    allocator::{Allocator, Buffer},
    dtype::DType,
};

use super::Device;

pub struct MetalDevice {
    raw_device: Arc<RawDevice>,
}

impl Default for MetalDevice {
    fn default() -> Self {
        Self {
            raw_device: Arc::new(
                RawDevice::system_default().expect("couldn't get system's default METAL device"),
            ),
        }
    }
}

impl Device for MetalDevice {
    type Allocator = MetalAllocator;

    fn allocator(&self) -> Self::Allocator {
        MetalAllocator {
            device: self.raw_device.clone(),
        }
    }
}

pub struct MetalBuffer<Dtype: DType> {
    raw_buffer: RawBuffer,
    slice: &'static mut [Dtype],
}

impl<Dtype: DType> Buffer<Dtype> for MetalBuffer<Dtype> {
    fn copy_in(&mut self, src: &[Dtype]) {
        assert_eq!(self.slice.len(), src.len());
        self.slice.copy_from_slice(src);
    }

    fn copy_out(&self, dst: &mut [Dtype]) {
        assert_eq!(dst.len(), self.slice.len());
        dst.copy_from_slice(self.slice);
    }
}

pub struct MetalAllocator {
    device: Arc<RawDevice>,
}

impl Allocator for MetalAllocator {
    type Buffer<Dtype: DType> = MetalBuffer<Dtype>;

    fn alloc<Dtype: DType>(&self, size: usize) -> Result<Self::Buffer<Dtype>, AllocError> {
        let raw_buffer = self.device.new_buffer(
            (size * std::mem::size_of::<Dtype>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        if raw_buffer.as_ptr().is_null() {
            // metal bindings api doesn't already return a result ...
            return Err(AllocError);
        }

        let slice = unsafe {
            std::slice::from_raw_parts_mut::<Dtype>(raw_buffer.contents() as *mut Dtype, size)
        };
        Ok(MetalBuffer { raw_buffer, slice })
    }

    unsafe fn free<Dtype: DType>(&self, b: Self::Buffer<Dtype>) {
        drop(b.raw_buffer) // explicit drop, hopefully release on drop :joy:
    }
}

#[cfg(test)]
mod test {
    use std::usize;

    use crate::{
        allocator::Allocator,
        backends::{metal::MetalDevice, Device},
    };

    #[test]
    fn oom() {
        let allocator = MetalDevice::default().allocator();
        assert!(allocator.alloc::<u8>(usize::MAX).is_err());
    }

    #[test]
    fn simple() {
        let allocator = MetalDevice::default().allocator();
        let _ = allocator.alloc::<f64>(16).unwrap();
    }
}
