use std::marker::PhantomData;

use metal::{Buffer as RawBuffer, Device as RawDevice, MTLResourceOptions};

use crate::{
    allocator::{Allocator, Buffer},
    dtype::DType,
};

// pub struct MetalDevice {
//     raw_device: Device,
// }

struct MetalBuffer<const N: usize, Dtype: DType> {
    raw_buffer: RawBuffer,
    dtype: PhantomData<Dtype>,
}

impl<const N: usize, Dtype: DType> Buffer<N, Dtype> for MetalBuffer<N, Dtype> {
    fn copy_in(&mut self, r#in: &[Dtype]) {
        todo!()
    }

    fn copy_out(&self, out: &mut [Dtype]) {
        todo!()
    }
}

struct MetalAllocator {
    device: RawDevice,
}

unsafe impl<const N: usize, Dtype: DType> Allocator<N, Dtype, MetalBuffer<N, Dtype>>
    for MetalAllocator
{
    fn alloc(&self) -> MetalBuffer<N, Dtype> {
        MetalBuffer {
            raw_buffer: self.device.new_buffer(
                (N * std::mem::size_of::<Dtype>()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            dtype: PhantomData,
        }
    }

    fn free(&self, b: MetalBuffer<N, Dtype>) {
        drop(b.raw_buffer);
    }
}
