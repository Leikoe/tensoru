fn main() {
    // Start by choosing a device to put the tensor on
    // we will use the CpuDevice in this little guide
    use tensoru::backends::CpuDevice;

    // Then we can use the [`Tensor`]
    use tensoru::Tensor;

    // A tensor is generic over it's DType and it's Device
    let _tensor_decl: Tensor<f32, CpuDevice>;

    // A tensor's shape is just an array ([`&[usize]`])
    let scalar_shape = [1];

    // Create an empty tensor by using [`Tensor::empty`]. The tensor will be allocated but unitialized.
    let _empty_tensor = Tensor::<f32, CpuDevice>::empty(&scalar_shape);

    // Create a zero intialized tensor by using [`Tensor::zeros`].
    let _zero_tensor = Tensor::<f32, CpuDevice>::zeros(&scalar_shape);
}
