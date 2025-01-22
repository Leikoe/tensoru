use crate::{
    backends::{CpuBuffer, Device},
    buffer::Buffer,
    compute_graph::{BinaryOpNode, LoadNode, Node},
    dtype::DType,
    op::{AddOp, BinaryOp},
};
use std::{fmt::Debug, marker::PhantomData, ops::Add};

#[derive(Debug)]
pub struct Tensor<DTYPE: DType, DEVICE: Device> {
    pub shape: Vec<usize>,
    data: Box<dyn Node<Dtype = DTYPE, Device = DEVICE>>,
    device: PhantomData<DEVICE>,
}

impl<DTYPE: DType, DEVICE: Device> Tensor<DTYPE, DEVICE> {
    pub fn empty(shape: &[usize]) -> Tensor<DTYPE, DEVICE> {
        let numel = shape.iter().product();
        Tensor {
            shape: shape.to_vec(),
            data: Box::new(LoadNode(DEVICE::Buffer::<DTYPE>::new(numel).unwrap())),
            device: PhantomData,
        }
    }

    pub fn zeros(shape: &[usize]) -> Tensor<DTYPE, DEVICE> {
        let numel = shape.iter().product();
        Tensor {
            shape: shape.to_vec(),
            data: Box::new(LoadNode(
                CpuBuffer::from(vec![DTYPE::ZERO; numel]).to::<DEVICE>(),
            )),
            device: PhantomData,
        }
    }

    pub fn from_slice(shape: &[usize], data: &[DTYPE]) -> Tensor<DTYPE, DEVICE> {
        let numel = shape.iter().product();
        assert_eq!(
            numel,
            data.len(),
            "provided data wasn't the right size for shape: {:?}",
            shape
        );

        let mut b = DEVICE::Buffer::<DTYPE>::new(numel).unwrap();
        b.copy_in(data);
        Tensor {
            shape: shape.to_vec(),
            data: Box::new(LoadNode(CpuBuffer::from(data).to::<DEVICE>())),
            device: PhantomData,
        }
    }
}

impl<DTYPE: DType, DEVICE: Device> Add<Tensor<DTYPE, DEVICE>> for Tensor<DTYPE, DEVICE> {
    type Output = Tensor<DTYPE, DEVICE>;

    fn add(self, rhs: Tensor<DTYPE, DEVICE>) -> Self::Output
    where
        AddOp: BinaryOp<DTYPE, DTYPE, DEVICE, Output = DTYPE>,
    {
        assert_eq!(self.shape, rhs.shape);
        let self_shape = self.shape.clone();
        let op: BinaryOpNode<AddOp, _, _, DEVICE> =
            BinaryOpNode(self.data, rhs.data, PhantomData, PhantomData);
        Tensor {
            shape: self_shape,
            data: Box::new(op),
            device: PhantomData,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{backends::CpuDevice, Tensor};

    #[test]
    fn test_add_graph() {
        let a = Tensor::<f32, CpuDevice>::zeros(&[1]);
        let b = Tensor::<f32, CpuDevice>::zeros(&[1]);
        let r = a + b;
        panic!("should have been an add of a and b");
    }
}
