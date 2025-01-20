use crate::{dtype::DType, utils::Prod};

#[derive(Debug, Clone)]
pub struct Tensor<const N: usize, Dtype: DType> {
    pub shape: [usize; N],
    pub data: Vec<Dtype>,
}

impl<const N: usize, Dtype: DType> Tensor<N, Dtype> {
    pub fn zeros(shape: [usize; N]) -> Tensor<N, Dtype> {
        let numel = shape.prod();
        assert_ne!(numel, 0, "cannot create a zero sized tensor");
        Tensor {
            shape,
            data: vec![Dtype::zero(); shape.prod()],
        }
    }

    pub fn with_data(shape: [usize; N], data: Vec<Dtype>) -> Tensor<N, Dtype> {
        let numel = shape.prod();
        assert_eq!(
            data.len(),
            numel,
            "provided data isn't the right size for given shape"
        );
        Tensor { shape, data }
    }

    pub fn from_slice(shape: [usize; N], data: &[Dtype]) -> Tensor<N, Dtype> {
        let numel = shape.prod();
        assert_eq!(
            data.len(),
            numel,
            "provided slice isn't the right size for given shape"
        );
        Tensor {
            shape,
            data: data.to_vec(),
        }
    }
}
