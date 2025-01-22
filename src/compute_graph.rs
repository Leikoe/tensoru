use std::{fmt::Debug, marker::PhantomData};

use crate::{
    backends::Device,
    dtype::DType,
    op::{BinaryOp, UnaryOp},
};

pub trait Node: Debug {
    type Dtype: DType;
    type Device: Device;
}

#[derive(Debug, Clone)]
pub struct LoadNode<DTYPE: DType, DEVICE: Device>(pub DEVICE::Buffer<DTYPE>);

impl<Dtype: DType, D: Device> Node for LoadNode<Dtype, D> {
    type Dtype = Dtype;
    type Device = D;
}

#[derive(Debug, Clone, Copy)]
pub struct UnaryOpNode<OP: UnaryOp<INPUT::Dtype, DEVICE>, INPUT: Node, DEVICE: Device>(
    INPUT,
    PhantomData<OP>,
    PhantomData<DEVICE>,
);

impl<OP: UnaryOp<INPUT::Dtype, DEVICE>, INPUT: Node, DEVICE: Device> Node
    for UnaryOpNode<OP, INPUT, DEVICE>
{
    type Dtype = OP::Output;
    type Device = INPUT::Device;
}

#[derive(Debug)]
pub struct BinaryOpNode<
    OP: BinaryOp<LHS::Dtype, RHS::Dtype, DEVICE>,
    LHS: Node,
    RHS: Node,
    DEVICE: Device,
>(
    pub LHS,
    pub RHS,
    pub PhantomData<OP>,
    pub PhantomData<DEVICE>,
);

impl<OP: BinaryOp<LHS::Dtype, RHS::Dtype, DEVICE>, LHS: Node, RHS: Node, DEVICE: Device> Node
    for BinaryOpNode<OP, LHS, RHS, DEVICE>
{
    type Dtype = OP::Output;
    type Device = LHS::Device;
}

impl<DTYPE: DType, DEVICE: Device> Node for Box<dyn Node<Dtype = DTYPE, Device = DEVICE>> {
    type Dtype = DTYPE;
    type Device = DEVICE;
}

#[cfg(test)]
mod test {
    use std::marker::PhantomData;

    use crate::{
        backends::{CpuBuffer, CpuDevice},
        buffer::Buffer,
        op::AbsOp,
    };

    use super::{LoadNode, UnaryOpNode};

    #[test]
    fn test_simple() {
        let a = CpuBuffer::new(16).unwrap();
        let l = LoadNode::<f32, CpuDevice>(a);
        let _ast = UnaryOpNode(l, PhantomData::<AbsOp>, PhantomData::<CpuDevice>);
    }
}
