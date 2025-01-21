#![feature(f16)]
#![feature(allocator_api)]
#![allow(refining_impl_trait)]
pub mod backends;
mod buffer;
mod compute_graph;
pub mod dtype;
pub mod op;
mod tensor;
mod utils;

pub use tensor::Tensor;

#[cfg(test)]
mod tests {
    use crate::{
        backends::{CpuDevice, MetalDevice},
        // op::{Add, BinaryOp},
        tensor::Tensor,
    };

    // #[test]
    // fn zeros() {
    //     let a: Tensor<f64, MetalDevice> = Tensor::zeros(&[3]);
    //     assert_eq!(a.to_vec(), vec![0.; 3]);
    // }

    // #[test]
    // fn ones() {
    //     let a: Tensor<f64, MetalDevice> = Tensor::from_slice(&[3], &[1., 1., 1.]);
    //     assert_eq!(a.to_vec(), vec![1.; 3]);
    // }

    // #[test]
    // fn add_to_zeros() {
    //     let a: Tensor<f64, CpuDevice> = Tensor::zeros(&[3]);
    //     let b: Tensor<f64, CpuDevice> = Tensor::from_slice(&[3], &[1., 2., 3.]);
    //     let result = Add::forward(a, b);
    //     assert_eq!(result.to_vec(), vec![1., 2., 3.]);
    // }

    // #[test]
    // fn add_zeros() {
    //     let b: Tensor<f64, CpuDevice> = Tensor::from_slice(&[3], &[1., 2., 3.]);
    //     let a: Tensor<f64, CpuDevice> = Tensor::zeros(&[3]);
    //     let result = Add::forward(a, b);
    //     assert_eq!(result.to_vec(), vec![1., 2., 3.]);
    // }
}
