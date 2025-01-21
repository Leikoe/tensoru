fn main() {
    // Start by choosing a device to put the tensor on
    // we will use the CpuDevice in this little guide
    use tensoru::backends::CpuDevice;

    // Then we can use the [`Tensor`]
    use tensoru::Tensor;

    // A tensor is generic over it's Rank, It's DType and It's Device
    let _tensor_decl: Tensor<2, f32, CpuDevice>;

    // Create an empty tensor by using [`Tensor::empty`]. The tensor will be allocated but unitialized.
    let _empty_tensor = Tensor::<1, f32, CpuDevice>::empty([1]);

    // Create a zero intialized tensor by using [`Tensor::zeros`].
    let _zero_tensor = Tensor::<1, f32, CpuDevice>::zeros([1]);
}
