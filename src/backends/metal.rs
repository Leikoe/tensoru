use metal::{
    foreign_types::ForeignType, Buffer as RawBuffer, Device as RawDevice, MTLResourceOptions,
};
use std::{alloc::AllocError, fmt::Debug, sync::Arc};

use crate::{
    allocator::{Allocator, Buffer},
    dtype::DType,
};

use super::{cpu::CpuDevice, Device, HasAllocator};

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

impl HasAllocator for MetalDevice {
    type Allocator<'device> = MetalAllocator<'device>;
}

impl Device for MetalDevice {
    fn allocator(&self) -> Self::Allocator<'_> {
        MetalAllocator { device: self }
    }
}

pub struct MetalBuffer<'device, Dtype: DType> {
    raw_buffer: RawBuffer,
    slice: &'static mut [Dtype],
    device: &'device MetalDevice,
}

impl<'device, Dtype: DType> Debug for MetalBuffer<'device, Dtype> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // SAFETY: we are copying the data in it right after
        let mut v = unsafe { Box::new_uninit_slice(self.len()).assume_init() };
        self.copy_out(&mut v[..]);
        v.fmt(f)
    }
}

impl<'device, Dtype: DType> Buffer<Dtype> for MetalBuffer<'device, Dtype> {
    fn len(&self) -> usize {
        self.slice.len()
    }

    fn copy_in(&mut self, src: &[Dtype]) {
        assert_eq!(self.slice.len(), src.len());
        self.slice.copy_from_slice(src);
    }

    fn copy_out(&self, dst: &mut [Dtype]) {
        assert_eq!(dst.len(), self.slice.len());
        dst.copy_from_slice(self.slice);
    }
}

impl<'device, Dtype: DType> Clone for MetalBuffer<'device, Dtype> {
    fn clone(&self) -> Self {
        self.device
            .allocator()
            .alloc::<Dtype>(self.len())
            .expect("couldn't alloc while cloning METAL buffer")
    }
}

pub struct MetalAllocator<'device> {
    device: &'device MetalDevice,
}

impl<'device> Allocator<'device> for MetalAllocator<'device> {
    type Buffer<Dtype: DType> = MetalBuffer<'device, Dtype>;

    fn alloc<Dtype: DType>(&self, size: usize) -> Result<Self::Buffer<Dtype>, AllocError> {
        let raw_buffer = self.device.raw_device.new_buffer(
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
        Ok(MetalBuffer {
            raw_buffer,
            slice,
            device: self.device,
        })
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
        let device = MetalDevice::default();
        let allocator = device.allocator();
        assert!(allocator.alloc::<u8>(usize::MAX).is_err());
    }

    #[test]
    fn simple() {
        let device = MetalDevice::default();
        let allocator = device.allocator();
        let _ = allocator.alloc::<f64>(16).unwrap();
    }
}
