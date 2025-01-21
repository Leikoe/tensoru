use crate::{backends::Device, dtype::DType, tensor::Tensor};

#[derive(Debug, Clone)]
pub enum Op<Dtype: DType, D: Device> {
    Addition(Value<Dtype, D>, Value<Dtype, D>),
    Substraction(Value<Dtype, D>, Value<Dtype, D>),
}

#[derive(Debug, Clone)]
pub enum Value<Dtype: DType, D: Device> {
    Op(Box<Op<Dtype, D>>),
    Const(Tensor<Dtype, D>),
    ConstScalar(Dtype),
}
