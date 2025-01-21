use crate::{backends::CpuDevice, dtype::DType, tensor::Tensor};

pub trait BinaryOp {
    fn forward<D: DType>(a: Tensor<D, CpuDevice>, b: Tensor<D, CpuDevice>) -> Tensor<D, CpuDevice>;
}

pub trait UnaryOp {
    fn forward<D: DType>(a: Tensor<D, CpuDevice>) -> Tensor<D, CpuDevice>;
}

// pub struct Add;

// impl BinaryOp for Add {
//     fn forward<Dtype: DType>(
//         a: Tensor<Dtype, CpuDevice>,
//         b: Tensor<Dtype, CpuDevice>,
//     ) -> Tensor<Dtype, CpuDevice> {
//         assert!(a.shape == b.shape);

//         let v: Vec<Dtype> = a
//             .to_vec()
//             .into_iter()
//             .zip(b.to_vec().into_iter())
//             .map(|(a, b)| a + b)
//             .collect();
//         Tensor::from_slice(&a.shape, &v)
//     }
// }
