use crate::dtype::DType;

pub unsafe trait Allocator<const N: usize, Dtype: DType, B: Buffer<N, Dtype>> {
    fn alloc(&self) -> B;
    fn free(&self, b: B);
}

pub trait Buffer<const N: usize, Dtype: DType> {
    fn copy_in(&mut self, r#in: &[Dtype]);
    fn copy_out(&self, out: &mut [Dtype]);
}
