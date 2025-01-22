use crate::{
    backends::{CpuBuffer, Device},
    buffer::Buffer,
    compute_graph::{LoadNode, Node},
    dtype::DType,
};
use std::{fmt::Debug, marker::PhantomData};

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

// impl<
//         DTYPE: DType,
//         DEVICE: Device,
//         LHS: TensorData<DTYPE, DEVICE>,
//         RHS: TensorData<DTYPE, DEVICE>,
//     > Add<Tensor<DTYPE, DEVICE, RHS>> for Tensor<DTYPE, DEVICE, LHS>
// {
//     type Output = Tensor<DTYPE, DEVICE, Lazy<BinaryOpNode<AddOp, LHS::NodeType, RHS::NodeType>>>;

//     fn add(self, rhs: Tensor<DTYPE, DEVICE, RHS>) -> Self::Output {
//         assert_eq!(self.shape, rhs.shape);
//         let self_shape = self.shape.clone();
//         Tensor {
//             shape: self_shape,
//             data: Lazy(BinaryOpNode(self.into(), rhs.into(), PhantomData)),
//             dtype: PhantomData,
//             device: PhantomData,
//         }
//     }
// }

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
