#![feature(f16)]
mod compute_graph;
pub mod dtype;
pub mod op;
pub mod tensor;
mod utils;

#[cfg(test)]
mod tests {
    use op::{Add, BinaryOp};
    use tensor::Tensor;

    use super::*;

    #[test]
    fn add_to_zeros() {
        let a: Tensor<1, f64> = Tensor::zeros([3]);
        let b: Tensor<1, f64> = Tensor::with_data([3], vec![1., 2., 3.]);
        let result = Add::forward(a.clone(), b.clone());
        assert_eq!(result.shape, a.shape);
        assert_eq!(result.shape, b.shape);
        assert_eq!(result.data, vec![1., 2., 3.]);
    }

    #[test]
    fn add_zeros() {
        let b: Tensor<1, f64> = Tensor::with_data([3], vec![1., 2., 3.]);
        let a: Tensor<1, f64> = Tensor::zeros([3]);
        let result = Add::forward(b.clone(), a.clone());
        assert_eq!(result.shape, a.shape);
        assert_eq!(result.shape, b.shape);
        assert_eq!(result.data, vec![1., 2., 3.]);
    }
}
