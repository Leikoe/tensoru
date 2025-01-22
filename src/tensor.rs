use crate::{
    backends::Device,
    buffer::Buffer,
    compute_graph::{BinaryOpNode, LoadNode, Node},
    dtype::DType,
    op::AddOp,
    utils::Prod,
};
use std::{fmt::Debug, marker::PhantomData, ops::Add};

pub trait TensorData<Dtype: DType, D: Device> {
    type Data;
    type NodeType: Node;
}

pub struct Evaluated<Dtype: DType, B: Buffer<Dtype>>(B, PhantomData<Dtype>);
impl<Dtype: DType, B: Buffer<Dtype>, D: Device> TensorData<Dtype, D> for Evaluated<Dtype, B> {
    type Data = B;
    type NodeType = LoadNode<Dtype, D>;
}
pub struct Lazy<G: Node>(G);
impl<G: Node, D: Device> TensorData<G::Dtype, D> for Lazy<G> {
    type Data = G;
    type NodeType = G;
}

impl<DTYPE: DType, DEVICE: Device> Into<LoadNode<DTYPE, DEVICE>>
    for Tensor<DTYPE, DEVICE, Evaluated<DTYPE, DEVICE::Buffer<DTYPE>>>
{
    fn into(self) -> LoadNode<DTYPE, DEVICE> {
        LoadNode::<DTYPE, DEVICE>(self)
    }
}

// impl<DEVICE: Device, G: Node> Into<G> for Tensor<G::Dtype, DEVICE, Lazy<G>> {
//     fn into(self) -> G {
//         self.data.0
//     }
// }

#[derive(Debug, Clone)]
pub struct Tensor<DTYPE: DType, DEVICE: Device, DATA: TensorData<DTYPE, DEVICE>> {
    pub shape: Vec<usize>,
    data: DATA,
    dtype: PhantomData<DTYPE>,
    device: PhantomData<DEVICE>,
}

impl<DTYPE: DType, DEVICE: Device, DATA: TensorData<DTYPE, DEVICE>> Tensor<DTYPE, DEVICE, DATA> {
    pub fn empty(
        shape: &[usize],
    ) -> Tensor<DTYPE, DEVICE, Evaluated<DTYPE, DEVICE::Buffer<DTYPE>>> {
        let numel = shape.prod();
        assert_ne!(numel, 0, "cannot create a zero sized tensor");

        Tensor {
            shape: shape.to_vec(),
            data: Evaluated(DEVICE::Buffer::<DTYPE>::new(numel).unwrap(), PhantomData),
            dtype: PhantomData,
            device: PhantomData,
        }
    }

    pub fn zeros(
        shape: &[usize],
    ) -> Tensor<DTYPE, DEVICE, Evaluated<DTYPE, DEVICE::Buffer<DTYPE>>> {
        let numel = shape.prod();
        assert_ne!(numel, 0, "cannot create a zero sized tensor");

        let mut t = Self::empty(shape);
        t.data.0.copy_in(&vec![DTYPE::zero(); numel]);
        t
    }

    pub fn from_slice(
        shape: &[usize],
        data: &[DTYPE],
    ) -> Tensor<DTYPE, DEVICE, Evaluated<DTYPE, DEVICE::Buffer<DTYPE>>> {
        let numel = shape.prod();
        assert_eq!(
            numel,
            data.len(),
            "provided data wasn't the right size for shape: {:?}",
            shape
        );

        let mut t = Self::empty(shape);
        t.data.0.copy_in(data);
        t
    }
}

impl<DEVICE: Device, G: Node> Tensor<G::Dtype, DEVICE, Lazy<G>> {
    pub fn to_vec(&self) -> Vec<G::Dtype> {
        // eval graph
        unimplemented!()
    }
}

impl<DTYPE: DType, DEVICE: Device> Tensor<DTYPE, DEVICE, Evaluated<DTYPE, DEVICE::Buffer<DTYPE>>> {
    pub fn to_vec(&self) -> Vec<DTYPE> {
        let mut v = vec![DTYPE::zero(); self.data.0.len()];
        self.data.0.copy_out(&mut v);
        v
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
