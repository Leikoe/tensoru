use super::Device;
use crate::{
    allocator::{Allocator, Buffer},
    dtype::DType,
};
use metal::{
    foreign_types::ForeignType, Buffer as RawBuffer, Device as RawDevice, MTLResourceOptions,
};
use std::{
    alloc::AllocError,
    fmt::Debug,
    sync::{LazyLock, Mutex},
};

static RAW_DEVICE: LazyLock<Mutex<RawDevice>> = LazyLock::new(|| {
    Mutex::new(RawDevice::system_default().expect("couldn't get system's default METAL device"))
});

pub struct MetalDevice;

impl Debug for MetalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("METAL")
    }
}

impl Device for MetalDevice {
    type Allocator = MetalAllocator;
}

pub struct MetalBuffer<Dtype: DType> {
    raw_buffer: RawBuffer,
    slice: &'static mut [Dtype],
}

impl<Dtype: DType> Debug for MetalBuffer<Dtype> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // SAFETY: we are copying the data in it right after
        let mut v = unsafe { Box::new_uninit_slice(self.len()).assume_init() };
        self.copy_out(&mut v[..]);
        v.fmt(f)
    }
}

impl<Dtype: DType> Buffer<Dtype> for MetalBuffer<Dtype> {
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

impl<'device, Dtype: DType> Clone for MetalBuffer<Dtype> {
    fn clone(&self) -> Self {
        MetalAllocator::alloc::<Dtype>(self.len())
            .expect("couldn't alloc while cloning METAL buffer")
    }
}

pub struct MetalAllocator;

impl Allocator for MetalAllocator {
    type Buffer<Dtype: DType> = MetalBuffer<Dtype>;

    fn alloc<Dtype: DType>(size: usize) -> Result<Self::Buffer<Dtype>, AllocError> {
        let raw_buffer = RAW_DEVICE
            .lock()
            .expect("METAL's RAW_DEVICE mutex was poisoned.")
            .new_buffer(
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

    unsafe fn free<Dtype: DType>(b: Self::Buffer<Dtype>) {
        drop(b.raw_buffer) // explicit drop, hopefully release on drop :joy:
    }
}

#[cfg(test)]
mod test {
    use std::{thread, time::Duration, usize};

    use crate::{allocator::Allocator, backends::metal::MetalAllocator};

    use super::RAW_DEVICE;

    #[test]
    fn test_oom() {
        assert!(MetalAllocator::alloc::<u8>(usize::MAX).is_err());
    }

    #[test]
    fn test_simple() {
        let _ = MetalAllocator::alloc::<f64>(16).unwrap();
    }

    #[test]
    fn test_free() {
        let before = RAW_DEVICE.lock().unwrap().current_allocated_size();
        let buff = MetalAllocator::alloc::<f64>(1).unwrap();
        let after = RAW_DEVICE.lock().unwrap().current_allocated_size();
        assert!(before < after);
        unsafe { MetalAllocator::free(buff) };
        thread::sleep(Duration::from_millis(50)); // wait for real dealloc on device
        let after_free = RAW_DEVICE.lock().unwrap().current_allocated_size();
        assert_eq!(after_free, before);
    }
}
