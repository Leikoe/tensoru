// use crate::{
//     allocator::Allocator,
//     backends::Device,
//     compute_graph::{Op, Value},
//     dtype::DType,
//     utils::Prod,
// };
// use std::ops::Add;

// #[derive(Debug, Clone)]
// pub enum TensorData<const N: usize, Dtype: DType, D: Device> {
//     Evaluated(<D::Allocator as Allocator>::Buffer<Dtype>),
//     Lazy(Box<Value<N, Dtype>>),
// }

// fn new_lazy_data<const N: usize, Dtype: DType>(
//     graph: Value<N, Dtype>,
// ) -> Box<TensorData<N, Dtype>> {
//     Box::new(TensorData::Lazy(graph))
// }

// impl<const N: usize, Dtype: DType> From<Vec<Dtype>> for Box<TensorData<N, Dtype>> {
//     fn from(value: Vec<Dtype>) -> Self {
//         Box::new(TensorData::Evaluated(value))
//     }
// }

// #[derive(Debug, Clone, PartialEq, Eq)]
// pub struct Tensor<const N: usize, Dtype: DType, D: Device + 'static> {
//     pub shape: [usize; N],
//     data: TensorData<N, Dtype, D>,
//     device: &'static D,
// }

// impl<const N: usize, Dtype: DType, D: Device> Tensor<N, Dtype, Device> {
//     pub fn zeros(shape: [usize; N]) -> Tensor<N, Dtype> {
//         let numel = shape.prod();
//         assert_ne!(numel, 0, "cannot create a zero sized tensor");
//         Tensor {
//             shape,
//             data: vec![Dtype::zero(); shape.prod()].into(),
//         }
//     }

//     pub fn with_data(shape: [usize; N], data: Vec<Dtype>) -> Tensor<N, Dtype> {
//         let numel = shape.prod();
//         assert_eq!(
//             data.len(),
//             numel,
//             "provided data isn't the right size for given shape"
//         );
//         Tensor {
//             shape,
//             data: data.into(),
//         }
//     }

//     pub fn from_slice(shape: [usize; N], data: &[Dtype]) -> Tensor<N, Dtype> {
//         let numel = shape.prod();
//         assert_eq!(
//             data.len(),
//             numel,
//             "provided slice isn't the right size for given shape"
//         );
//         Tensor {
//             shape,
//             data: data.to_vec().into(),
//         }
//     }

//     pub fn to_vec(&self) -> Vec<Dtype> {
//         match self.data.as_ref() {
//             TensorData::Evaluated(vec) => vec.clone(),
//             TensorData::Lazy(value) => {
//                 if let TensorData::Evaluated(v) = value.eval_cpu().data.as_ref() {
//                     v.clone()
//                 } else {
//                     unreachable!();
//                 }
//             }
//         }
//     }
// }

// impl<const N: usize, Dtype: DType> Add<Tensor<N, Dtype>> for Tensor<N, Dtype> {
//     type Output = Self;

//     fn add(self, rhs: Tensor<N, Dtype>) -> Self::Output {
//         assert_eq!(self.shape, rhs.shape);
//         let self_shape = self.shape;
//         let addition = Box::new(Op::Addition(Value::Const(self), Value::Const(rhs)));
//         Tensor {
//             shape: self_shape,
//             data: new_lazy_data(Value::Op(addition)),
//         }
//     }
// }
