// use crate::{dtype::DType, tensor::Tensor};

// pub trait BinaryOp {
//     fn forward<const N: usize, D: DType>(a: Tensor<N, D>, b: Tensor<N, D>) -> Tensor<N, D>;
// }

// pub trait UnaryOp {
//     fn forward<const N: usize, D: DType>(a: Tensor<N, D>) -> Tensor<N, D>;
// }

// pub struct Add;

// impl BinaryOp for Add {
//     fn forward<const N: usize, D: DType>(a: Tensor<N, D>, b: Tensor<N, D>) -> Tensor<N, D> {
//         assert!(a.shape == b.shape);

//         Tensor::with_data(
//             a.shape,
//             a.to_vec()
//                 .into_iter()
//                 .zip(b.to_vec().into_iter())
//                 .map(|(a, b)| a + b)
//                 .collect(),
//         )
//     }
// }
