use tracing::debug;

use super::Device;
use crate::{buffer::Buffer, dtype::DType};
use std::{any::type_name, fmt::Debug};

#[derive(Copy, Clone)]
pub struct CpuDevice;

impl Device for CpuDevice {
    type Buffer<Dtype: DType> = CpuBuffer<Dtype>;
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

impl<Dtype: DType> Drop for CpuBuffer<Dtype> {
    fn drop(&mut self) {
        debug!(
            "dropped {}(len={})",
            type_name::<Self>().split("::").last().unwrap(),
            self.len()
        );
    }
}

impl<DTYPE: DType, T: Into<Box<[DTYPE]>>> From<T> for CpuBuffer<DTYPE> {
    fn from(value: T) -> Self {
        CpuBuffer {
            raw_buffer: value.into(),
        }
    }
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

    fn new(size: usize) -> Result<Self, std::alloc::AllocError> {
        // SAFETY: data has to be copied in the [`Bufffer`] before using it
        let raw_buffer: Box<[Dtype]> =
            unsafe { Box::<[Dtype]>::try_new_uninit_slice(size)?.assume_init() };
        Ok(CpuBuffer { raw_buffer })
    }
}

#[cfg(test)]
mod test {
    use std::usize;

    use crate::{backends::cpu::CpuBuffer, buffer::Buffer};

    #[test]
    fn oom() {
        assert!(CpuBuffer::<u8>::new(usize::MAX).is_err());
    }

    #[test]
    fn simple() {
        let _ = CpuBuffer::<f64>::new(16).unwrap();
    }
}
