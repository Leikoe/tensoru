#![feature(f16)]
#![feature(allocator_api)]
#![allow(refining_impl_trait)]
mod allocator;
mod backends;
mod compute_graph;
pub mod dtype;
pub mod op;
pub mod tensor;
mod utils;

#[cfg(test)]
mod tests {
    use crate::{
        backends::cpu::CpuDevice,
        op::{Add, BinaryOp},
        tensor::Tensor,
    };

    #[test]
    fn zeros() {
        let a: Tensor<1, f64, CpuDevice> = Tensor::zeros([3]);
        assert_eq!(a.to_vec(), vec![0.; 3]);
    }

    #[test]
    fn add_to_zeros() {
        let a: Tensor<1, f64, CpuDevice> = Tensor::zeros([3]);
        let b: Tensor<1, f64, CpuDevice> = Tensor::from_slice([3], &[1., 2., 3.]);
        let result = Add::forward(a, b);
        assert_eq!(result.to_vec(), vec![1., 2., 3.]);
    }

    #[test]
    fn add_zeros() {
        let b: Tensor<1, f64, CpuDevice> = Tensor::from_slice([3], &[1., 2., 3.]);
        let a: Tensor<1, f64, CpuDevice> = Tensor::zeros([3]);
        let result = Add::forward(a, b);
        assert_eq!(result.to_vec(), vec![1., 2., 3.]);
    }
}
