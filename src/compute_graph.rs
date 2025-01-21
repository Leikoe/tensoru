use crate::{backends::Device, dtype::DType, tensor::Tensor};

#[derive(Debug, Clone)]
pub enum Op<Dtype: DType, D: Device> {
    Addition(Value<Dtype, D>, Value<Dtype, D>),
    Substraction(Value<Dtype, D>, Value<Dtype, D>),
}

// impl<const N: usize, Dtype: DType> Op<N, Dtype> {
//     fn eval_cpu(&self) -> Tensor<N, Dtype> {
//         match self {
//             Op::Addition(a, b) => Add::forward(a.eval_cpu(), b.eval_cpu()),
//         }
//     }
// }

#[derive(Debug, Clone)]
pub enum Value<Dtype: DType, D: Device> {
    Op(Box<Op<Dtype, D>>),
    Const(Tensor<Dtype, D>),
    ConstScalar(Dtype),
}

// impl<Dtype: DType, D: Device> Value<Dtype, D> {
// fn const_scalar(v: Dtype) -> Value<Dtype> {
//     Value::ConstScalar(v)
// }

// pub fn eval_cpu(&self) -> Tensor<Dtype, D> {
//     match self {
//         Value::Op(op) => op.eval_cpu(),
//         Value::Const(tensor) => tensor.clone(),
//         Value::ConstScalar(s) => Tensor::from_slice([1; N], &[*s; 1]),
//     }
// }
// }

// #[cfg(test)]
// mod test {
//     use super::Value;

//     #[test]
//     fn test_simple_graph() {
//         let g = Value::Op()
//     }
// }
