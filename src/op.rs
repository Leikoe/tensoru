use crate::{backends::CpuDevice, dtype::DType, tensor::Tensor};

pub trait BinaryOp {
    fn forward<const N: usize, D: DType>(
        a: Tensor<N, D, CpuDevice>,
        b: Tensor<N, D, CpuDevice>,
    ) -> Tensor<N, D, CpuDevice>;
}

pub trait UnaryOp {
    fn forward<const N: usize, D: DType>(a: Tensor<N, D, CpuDevice>) -> Tensor<N, D, CpuDevice>;
}

pub struct Add;

impl BinaryOp for Add {
    fn forward<const N: usize, Dtype: DType>(
        a: Tensor<N, Dtype, CpuDevice>,
        b: Tensor<N, Dtype, CpuDevice>,
    ) -> Tensor<N, Dtype, CpuDevice> {
        assert!(a.shape == b.shape);

        let v: Vec<Dtype> = a
            .to_vec()
            .into_iter()
            .zip(b.to_vec().into_iter())
            .map(|(a, b)| a + b)
            .collect();
        Tensor::from_slice(a.shape, &v)
    }
}
