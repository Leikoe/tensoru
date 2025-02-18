use crate::{backends::Device, buffer::Buffer, dtype::DType};
use std::{fmt::Debug, sync::Arc};

#[derive(Clone, Debug)]
pub enum TensorData<T: DType, D: Device> {
    Realized(Arc<D::Buffer<T>>),
    UnRealized, // (Graph ???) the tensor should contain some graph
}

#[derive(Clone, Debug)]
pub struct Tensor<T: DType, D: Device> {
    pub shape: Vec<usize>,
    data: TensorData<T, D>,
}

impl<T: DType, D: Device> Tensor<T, D> {
    pub fn empty(shape: &[usize]) -> Self {
        Tensor {
            shape: shape.to_vec(),
            data: TensorData::UnRealized,
        }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let size = shape.iter().product();
        let mut buff = D::Buffer::<T>::new(size).expect("OOM");
        let zero_host_buff = vec![T::ZERO; size];
        buff.copy_in(&zero_host_buff);
        Tensor {
            shape: shape.to_vec(),
            data: TensorData::Realized(Arc::new(buff)),
        }
    }

    pub fn from_slice(shape: &[usize], data: &[T]) -> Self {
        let size = shape.iter().product();
        assert_eq!(size, data.len(), "mismatched shape and slice size");
        let mut buff = D::Buffer::<T>::new(size).expect("OOM");
        buff.copy_in(data);
        Tensor {
            shape: shape.to_vec(),
            data: TensorData::Realized(Arc::new(buff)),
        }
    }

    pub fn to_vec(self) -> Vec<T> {
        match self.data {
            TensorData::Realized(buff) => {
                let mut v = vec![T::ZERO; self.shape.iter().product()];
                buff.copy_out(v.as_mut_slice());
                v
            }
            TensorData::UnRealized => unimplemented!(),
        }
    }
}
