use crate::{
    backends::{cpu::CpuDevice, Device},
    buffer::Buffer,
    // compute_graph::{Op, Value},
    dtype::DType,
    utils::Prod,
};
use std::ops::Add;
use std::{fmt::Debug, marker::PhantomData};

pub enum TensorData<const N: usize, Dtype: DType, D: Device> {
    Evaluated(D::Buffer<Dtype>),
    // Lazy(Box<Value<N, Dtype>>),
}

impl<const N: usize, Dtype: DType, D: Device> Debug for TensorData<N, Dtype, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorData::Evaluated(buffer) => buffer.fmt(f),
            // TensorData::Lazy(value) => value.fmt(f),
        }
    }
}

impl<const N: usize, Dtype: DType, D: Device> Clone for TensorData<N, Dtype, D> {
    fn clone(&self) -> Self {
        match self {
            TensorData::Evaluated(buffer) => TensorData::Evaluated(buffer.clone()),
            // TensorData::Lazy(value) => TensorData::Lazy(value.clone()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tensor<const N: usize, Dtype: DType, D: Device> {
    pub shape: [usize; N],
    data: TensorData<N, Dtype, D>,
}

impl<'device, const N: usize, Dtype: DType, D: Device> Tensor<N, Dtype, D> {
    pub fn empty(shape: [usize; N]) -> Tensor<N, Dtype, D> {
        let numel = shape.prod();
        assert_ne!(numel, 0, "cannot create a zero sized tensor");

        Tensor {
            shape,
            data: TensorData::Evaluated(D::Buffer::<Dtype>::new(numel).unwrap()),
        }
    }

    pub fn zeros(shape: [usize; N]) -> Tensor<N, Dtype, D> {
        let numel = shape.prod();
        assert_ne!(numel, 0, "cannot create a zero sized tensor");

        let mut t = Self::empty(shape);
        if let TensorData::Evaluated(buff) = &mut t.data {
            buff.copy_in(&vec![Dtype::zero(); numel]);
        } else {
            unreachable!();
        }
        t
    }

    pub fn from_slice(shape: [usize; N], data: &[Dtype]) -> Tensor<N, Dtype, D> {
        let numel = shape.prod();
        assert_eq!(
            numel,
            data.len(),
            "provided data wasn't the right size for shape: {:?}",
            shape
        );

        let mut t = Self::empty(shape);
        if let TensorData::Evaluated(buff) = &mut t.data {
            buff.copy_in(data);
        } else {
            unreachable!();
        }
        t
    }

    pub fn to_vec(&self) -> Vec<Dtype> {
        match &self.data {
            TensorData::Evaluated(buff) => {
                let mut v = vec![Dtype::zero(); buff.len()];
                buff.copy_out(&mut v);
                v
            } // TensorData::Lazy(value) => value.eval_cpu().to_vec(),
        }
    }
}

// impl<const N: usize, Dtype: DType> Add<Tensor<N, Dtype, CpuDevice>>
//     for Tensor<N, Dtype, CpuDevice>
// {
//     type Output = Self;

//     fn add(self, rhs: Tensor<N, Dtype, CpuDevice>) -> Self::Output {
//         assert_eq!(self.shape, rhs.shape);
//         let self_shape = self.shape;
//         let addition = Box::new(Op::Addition(Value::Const(self), Value::Const(rhs)));
//         Tensor {
//             shape: self_shape,
//             data: new_lazy_data(Value::Op(addition)),
//         }
//     }
// }
