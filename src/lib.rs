#![feature(f16)]
#![feature(allocator_api)]
mod allocator;
mod backends;
mod compute_graph;
pub mod dtype;
pub mod op;
pub mod tensor;
mod utils;

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;

    #[test]
    fn playground() {}

    #[test]
    fn add_to_zeros() {
        let a: Tensor<1, f64> = Tensor::zeros([3]);
        let b: Tensor<1, f64> = Tensor::with_data([3], vec![1., 2., 3.]);
        let result = a + b;
        assert_eq!(result.to_vec(), vec![1., 2., 3.]);
    }

    #[test]
    fn add_zeros() {
        let b: Tensor<1, f64> = Tensor::with_data([3], vec![1., 2., 3.]);
        let a: Tensor<1, f64> = Tensor::zeros([3]);
        let result = a + b;
        assert_eq!(result.to_vec(), vec![1., 2., 3.]);
    }
}
