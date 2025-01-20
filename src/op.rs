use crate::{dtype::DType, tensor::Tensor};

pub trait BinaryOp {
    fn forward<const N: usize, D: DType>(a: Tensor<N, D>, b: Tensor<N, D>) -> Tensor<N, D>;
}

pub trait UnaryOp {
    fn forward<const N: usize, D: DType>(a: Tensor<N, D>) -> Tensor<N, D>;
}

pub struct Add;

impl BinaryOp for Add {
    fn forward<const N: usize, D: DType>(a: Tensor<N, D>, b: Tensor<N, D>) -> Tensor<N, D> {
        assert!(a.shape == b.shape);
        Tensor {
            shape: a.shape,
            data: a
                .data
                .into_iter()
                .zip(b.data.into_iter())
                .map(|(a, b)| a + b)
                .collect(),
        }
    }
}
