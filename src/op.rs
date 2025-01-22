use std::fmt::Debug;

use crate::{backends::Device, dtype::DType, tensor::Tensor};

pub trait Op: Debug + 'static + Copy {}

pub trait UnaryOp<IDTYPE: DType, DEVICE: Device>: Op {
    type Output: DType;
    fn forward(a: Tensor<IDTYPE, DEVICE>) -> Tensor<Self::Output, DEVICE>;
}

pub trait BinaryOp<LHSDTYPE: DType, RHSDTYPE: DType, DEVICE: Device>: Op {
    type Output: DType;
    fn forward(
        lhs: Tensor<LHSDTYPE, DEVICE>,
        rhs: Tensor<RHSDTYPE, DEVICE>,
    ) -> Tensor<Self::Output, DEVICE>;
}

#[derive(Debug, Clone, Copy)]
pub struct AbsOp;
impl Op for AbsOp {}
impl<IDTYPE: DType, DEVICE: Device> UnaryOp<IDTYPE, DEVICE> for AbsOp {
    type Output = IDTYPE;

    fn forward(_a: Tensor<IDTYPE, DEVICE>) -> Tensor<Self::Output, DEVICE> {
        unimplemented!()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AddOp;
impl Op for AddOp {}
impl<IDTYPE: DType, DEVICE: Device> BinaryOp<IDTYPE, IDTYPE, DEVICE> for AddOp {
    type Output = IDTYPE;

    fn forward(
        _lhs: Tensor<IDTYPE, DEVICE>,
        _rhs: Tensor<IDTYPE, DEVICE>,
    ) -> Tensor<Self::Output, DEVICE> {
        unimplemented!()
    }
}

/// Marker trait for Associative operations
pub trait Associative {}
