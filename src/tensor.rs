use crate::{
    backends::Device,
    buffer::Buffer,
    compute_graph::{Op, Value},
    dtype::DType,
    utils::Prod,
};
use std::{fmt::Debug, ops::Add};

#[derive(Debug, Clone)]
pub enum TensorData<Dtype: DType, D: Device> {
    Evaluated(D::Buffer<Dtype>),
    Lazy(Box<Value<Dtype, D>>),
}

#[derive(Debug, Clone)]
pub struct Tensor<Dtype: DType, D: Device> {
    pub shape: Vec<usize>,
    data: TensorData<Dtype, D>,
}

impl<Dtype: DType, D: Device> Tensor<Dtype, D> {
    pub fn empty(shape: &[usize]) -> Tensor<Dtype, D> {
        let numel = shape.prod();
        assert_ne!(numel, 0, "cannot create a zero sized tensor");

        Tensor {
            shape: shape.to_vec(),
            data: TensorData::Evaluated(D::Buffer::<Dtype>::new(numel).unwrap()),
        }
    }

    pub fn zeros(shape: &[usize]) -> Tensor<Dtype, D> {
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

    pub fn from_slice(shape: &[usize], data: &[Dtype]) -> Tensor<Dtype, D> {
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

    // pub fn to_vec(&self) -> Vec<Dtype> {
    //     match &self.data {
    //         TensorData::Evaluated(buff) => {
    //             let mut v = vec![Dtype::zero(); buff.len()];
    //             buff.copy_out(&mut v);
    //             v
    //         }
    //         TensorData::Lazy(value) => value.eval_cpu().to_vec(),
    //     }
    // }
}

impl<Dtype: DType, D: Device> Add<Tensor<Dtype, D>> for Tensor<Dtype, D> {
    type Output = Self;

    fn add(self, rhs: Tensor<Dtype, D>) -> Self::Output {
        assert_eq!(self.shape, rhs.shape);
        let self_shape = self.shape.clone();
        let addition = Box::new(Op::Addition(Value::Const(self), Value::Const(rhs)));
        Tensor {
            shape: self_shape,
            data: TensorData::Lazy(Box::new(Value::Op(addition))),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{backends::CpuDevice, compute_graph::Op};

    use super::{Tensor, TensorData, Value};

    #[test]
    fn test_add_graph() {
        let a = Tensor::<f32, CpuDevice>::zeros(&[1]);
        let b = Tensor::<f32, CpuDevice>::zeros(&[1]);
        let r = a + b;
        if let TensorData::Lazy(v) = r.data {
            if let Value::Op(o) = *v {
                if let Op::Addition(_, _) = *o {
                    return;
                }
            }
        }
        panic!("should have been an add of a and b");
    }
}
