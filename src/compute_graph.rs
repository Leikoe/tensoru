// use crate::{
//     dtype::DType,
//     op::{Add, BinaryOp},
//     tensor::Tensor,
// };

// #[derive(Debug, PartialEq, Eq, Clone)]
// pub enum Op<const N: usize, Dtype: DType> {
//     Addition(Value<N, Dtype>, Value<N, Dtype>),
// }

// impl<const N: usize, Dtype: DType> Op<N, Dtype> {
//     fn eval_cpu(&self) -> Tensor<N, Dtype> {
//         match self {
//             Op::Addition(a, b) => Add::forward(a.eval_cpu(), b.eval_cpu()),
//         }
//     }
// }

// #[derive(Debug, PartialEq, Eq, Clone)]
// pub enum Value<const N: usize, Dtype: DType> {
//     Op(Box<Op<N, Dtype>>),
//     Const(Tensor<N, Dtype>),
//     ConstScalar(Dtype),
// }

// impl<const N: usize, Dtype: DType> Value<N, Dtype> {
//     fn const_scalar(v: Dtype) -> Value<N, Dtype> {
//         Value::ConstScalar(v)
//     }

//     pub fn eval_cpu(&self) -> Tensor<N, Dtype> {
//         match self {
//             Value::Op(op) => op.eval_cpu(),
//             Value::Const(tensor) => tensor.clone(),
//             Value::ConstScalar(s) => Tensor::with_data([1; N], vec![*s; 1]),
//         }
//     }
// }

// #[cfg(test)]
// mod test {
//     use super::*;

//     #[test]
//     fn simple_graph() {
//         let graph = Value::Op(Box::new(Op::Addition(
//             Value::<1, f64>::const_scalar(1.),
//             Value::<1, f64>::const_scalar(2.),
//         )));

//         assert_eq!(graph.eval_cpu(), Tensor::from_slice([1], &[3.]));
//     }
// }
