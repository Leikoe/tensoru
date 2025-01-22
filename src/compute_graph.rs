use std::marker::PhantomData;

use crate::{
    backends::Device,
    dtype::DType,
    op::{BinaryOp, UnaryOp},
    tensor::Evaluated,
    Tensor,
};

pub trait Node {
    type Dtype: DType;
}

pub struct LoadNode<DTYPE: DType, DEVICE: Device>(
    Tensor<DTYPE, DEVICE, Evaluated<DTYPE, DEVICE::Buffer<DTYPE>>>,
);

impl<Dtype: DType, D: Device> Node for LoadNode<Dtype, D> {
    type Dtype = Dtype;
}

pub struct UnaryOpNode<OP: UnaryOp<INPUT::Dtype>, INPUT: Node>(INPUT, PhantomData<OP>);

impl<OP: UnaryOp<INPUT::Dtype>, INPUT: Node> Node for UnaryOpNode<OP, INPUT> {
    type Dtype = OP::Output;
}

pub struct BinaryOpNode<OP: BinaryOp<LHS::Dtype, RHS::Dtype>, LHS: Node, RHS: Node>(
    LHS,
    RHS,
    PhantomData<OP>,
);

impl<OP: BinaryOp<LHS::Dtype, RHS::Dtype>, LHS: Node, RHS: Node> Node
    for BinaryOpNode<OP, LHS, RHS>
{
    type Dtype = OP::Output;
}

#[cfg(test)]
mod test {
    use std::marker::PhantomData;

    use crate::{
        backends::{CpuBuffer, CpuDevice},
        buffer::Buffer,
        op::Abs,
        tensor::Evaluated,
        Tensor,
    };

    use super::{LoadNode, UnaryOpNode};

    #[test]
    fn test_simple() {
        let a = Tensor::<f32, CpuDevice, Evaluated<f32, CpuBuffer<f32>>>::zeros(&[3]);
        let l = LoadNode::<f32, CpuDevice>(a);
        let ast = UnaryOpNode(l, PhantomData::<Abs>);
    }
}
