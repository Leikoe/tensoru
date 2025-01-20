use crate::allocator::Allocator;

pub mod cpu;
#[cfg(target_os = "macos")]
pub mod metal;

pub trait Device {
    type Allocator: Allocator;
    fn allocator(&self) -> Self::Allocator;
}
