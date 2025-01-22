use crate::{
    backends::Device,
    dtype::DType,
    tensor::{Tensor, TensorData},
};

pub trait UnaryOp<IDTYPE: DType> {
    type Output: DType;
    fn forward<
        DEVICE: Device,
        IDATA: TensorData<IDTYPE, DEVICE>,
        ODATA: TensorData<Self::Output, DEVICE>,
    >(
        a: Tensor<IDTYPE, DEVICE, IDATA>,
    ) -> Tensor<Self::Output, DEVICE, ODATA>;
}

pub trait BinaryOp<LHSDTYPE: DType, RHSDTYPE: DType> {
    type Output: DType;
    fn forward<
        DEVICE: Device,
        LHSDATA: TensorData<LHSDTYPE, DEVICE>,
        RHSDATA: TensorData<RHSDTYPE, DEVICE>,
        ODATA: TensorData<Self::Output, DEVICE>,
    >(
        lhs: Tensor<LHSDTYPE, DEVICE, LHSDATA>,
        rhs: Tensor<RHSDTYPE, DEVICE, RHSDATA>,
    ) -> Tensor<Self::Output, DEVICE, ODATA>;
}

pub struct AbsOp;

impl<INPUT: DType> UnaryOp<INPUT> for AbsOp {
    type Output = INPUT;

    fn forward<
        DEVICE: Device,
        IDATA: TensorData<INPUT, DEVICE>,
        ODATA: TensorData<Self::Output, DEVICE>,
    >(
        _a: Tensor<INPUT, DEVICE, IDATA>,
    ) -> Tensor<Self::Output, DEVICE, ODATA> {
        unimplemented!()
    }
}

pub struct AddOp;

impl<INPUT: DType> BinaryOp<INPUT, INPUT> for AddOp {
    type Output = INPUT;

    fn forward<
        DEVICE: Device,
        LHSDATA: TensorData<INPUT, DEVICE>,
        RHSDATA: TensorData<INPUT, DEVICE>,
        ODATA: TensorData<Self::Output, DEVICE>,
    >(
        _lhs: Tensor<INPUT, DEVICE, LHSDATA>,
        _rhs: Tensor<INPUT, DEVICE, RHSDATA>,
    ) -> Tensor<Self::Output, DEVICE, ODATA> {
        unimplemented!()
    }
}

/// Marker trait for Associative operations
pub trait Associative {}
