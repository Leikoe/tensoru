use crate::allocator::Allocator;

pub mod cpu;
#[cfg(target_os = "macos")]
pub mod metal;

trait HasAllocator {
    type Allocator<'device>: Allocator<'device>;
}

pub trait Device: HasAllocator {
    fn allocator(&self) -> Self::Allocator<'_>;
}
