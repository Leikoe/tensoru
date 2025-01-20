use crate::{
    allocator::{Allocator, Buffer},
    backends::Device,
    compute_graph::{Op, Value},
    dtype::DType,
    utils::Prod,
};
use std::fmt::Debug;
use std::ops::Add;

pub enum TensorData<'device, const N: usize, Dtype: DType, D: Device> {
    Evaluated(<D::Allocator<'device> as Allocator<'device>>::Buffer<Dtype>),
    Lazy(Box<Value<N, Dtype>>),
}

impl<'device, const N: usize, Dtype: DType, D: Device> Debug for TensorData<'device, N, Dtype, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorData::Evaluated(buffer) => buffer.fmt(f),
            TensorData::Lazy(value) => value.fmt(f),
        }
    }
}

impl<'device, const N: usize, Dtype: DType, D: Device> Clone for TensorData<'device, N, Dtype, D> {
    fn clone(&self) -> Self {
        match self {
            TensorData::Evaluated(buffer) => TensorData::Evaluated(buffer.clone()),
            TensorData::Lazy(value) => TensorData::Lazy(value.clone()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tensor<'device, const N: usize, Dtype: DType, D: Device> {
    pub shape: [usize; N],
    data: TensorData<'device, N, Dtype, D>,
    device: &'device D,
}

impl<'device, const N: usize, Dtype: DType, D: Device> Tensor<'device, N, Dtype, D> {
    pub fn zeros(shape: [usize; N]) -> Tensor<'device, N, Dtype, D> {
        let numel = shape.prod();
        assert_ne!(numel, 0, "cannot create a zero sized tensor");
        Tensor {
            shape,
            data: ,
        }
    }

    pub fn with_data(shape: [usize; N], data: Vec<Dtype>) -> Tensor<N, Dtype> {
        let numel = shape.prod();
        assert_eq!(
            data.len(),
            numel,
            "provided data isn't the right size for given shape"
        );
        Tensor {
            shape,
            data: data.into(),
        }
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
            data: data.to_vec().into(),
        }
    }

    pub fn to_vec(&self) -> Vec<Dtype> {
        match self.data.as_ref() {
            TensorData::Evaluated(vec) => vec.clone(),
            TensorData::Lazy(value) => {
                if let TensorData::Evaluated(v) = value.eval_cpu().data.as_ref() {
                    v.clone()
                } else {
                    unreachable!();
                }
            }
        }
    }
}

impl<const N: usize, Dtype: DType> Add<Tensor<N, Dtype>> for Tensor<N, Dtype> {
    type Output = Self;

    fn add(self, rhs: Tensor<N, Dtype>) -> Self::Output {
        assert_eq!(self.shape, rhs.shape);
        let self_shape = self.shape;
        let addition = Box::new(Op::Addition(Value::Const(self), Value::Const(rhs)));
        Tensor {
            shape: self_shape,
            data: new_lazy_data(Value::Op(addition)),
        }
    }
}
