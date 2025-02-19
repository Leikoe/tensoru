use super::Device;
use crate::{buffer::Buffer, dtype::DType};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLBuffer, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions};
use std::{
    alloc::AllocError,
    any::type_name,
    fmt::Debug,
    ops::Deref,
    sync::{Arc, LazyLock, Mutex},
};
use tracing::debug;

// Linking to CoreGraphics to use MTLCreateSystemDefaultDevice
#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {}

// This is only because [`Retained<ProtocolObject<dyn MTLDevice>>`] doesn't implement Send and Sync even though it should.
struct MTLDeviceWrapper(Retained<ProtocolObject<dyn MTLDevice>>);

unsafe impl Send for MTLDeviceWrapper {}
unsafe impl Sync for MTLDeviceWrapper {}

impl Deref for MTLDeviceWrapper {
    type Target = Retained<ProtocolObject<dyn MTLDevice>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

static RAW_DEVICE: LazyLock<Mutex<MTLDeviceWrapper>> = LazyLock::new(|| {
    Mutex::new(MTLDeviceWrapper(
        MTLCreateSystemDefaultDevice().expect("couldn't get system's default METAL device"),
    ))
});

#[derive(Copy, Clone)]
pub struct MetalDevice;

impl Debug for MetalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("METAL")
    }
}

impl Device for MetalDevice {
    type Buffer<Dtype: DType> = MetalBuffer<Dtype>;
}

pub struct MetalBuffer<Dtype: DType> {
    _raw_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
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

impl<Dtype: DType> Drop for MetalBuffer<Dtype> {
    fn drop(&mut self) {
        debug!(
            "dropped {}(len={})",
            type_name::<Self>().split("::").last().unwrap(),
            self.len()
        );
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

    fn new(size: usize) -> Result<Self, AllocError> {
        let raw_buffer = RAW_DEVICE
            .lock()
            .expect("METAL's RAW_DEVICE mutex was poisoned.")
            .newBufferWithLength_options(
                size * std::mem::size_of::<Dtype>(),
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or(AllocError)?;
        let slice = unsafe {
            std::slice::from_raw_parts_mut::<Dtype>(
                raw_buffer.contents().as_ptr() as *mut Dtype,
                size,
            )
        };
        Ok(MetalBuffer {
            _raw_buffer: raw_buffer,
            slice,
        })
    }
}

impl<'device, Dtype: DType> Clone for MetalBuffer<Dtype> {
    fn clone(&self) -> Self {
        let mut new_buff = MetalBuffer::<Dtype>::new(self.len())
            .expect("couldn't alloc while cloning METAL buffer");
        new_buff.copy_in(self.slice);
        new_buff
    }
}

#[cfg(test)]
mod test {
    use std::{thread, time::Duration, usize};

    use tracing_test::traced_test;

    use crate::{backends::metal::MetalBuffer, buffer::Buffer};

    use super::RAW_DEVICE;

    #[test]
    #[traced_test]
    fn test_oom() {
        assert!(MetalBuffer::<u8>::new(usize::MAX).is_err());
    }

    #[test]
    #[traced_test]
    fn test_simple() {
        let _ = MetalBuffer::<f64>::new(16).unwrap();
    }

    #[test]
    #[traced_test]
    fn test_free() {
        let before = RAW_DEVICE.lock().unwrap().current_allocated_size();
        let buff = MetalBuffer::<f64>::new(10_usize.pow(9)).unwrap();
        let after = RAW_DEVICE.lock().unwrap().current_allocated_size();
        assert!(before < after);
        drop(buff);
        thread::sleep(Duration::from_millis(50)); // wait for real dealloc on device
        let after_free = RAW_DEVICE.lock().unwrap().current_allocated_size();
        assert_eq!(after_free, before);
    }
}
