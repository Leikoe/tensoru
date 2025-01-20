use crate::allocator::Allocator;
use std::fmt::Debug;

pub mod cpu;
#[cfg(target_os = "macos")]
pub mod metal;

pub trait Device: Debug {
    type Allocator: Allocator;
}
