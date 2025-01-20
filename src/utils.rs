pub trait Prod {
    fn prod(&self) -> usize;
}

impl<const N: usize> Prod for [usize; N] {
    fn prod(&self) -> usize {
        self.iter().copied().reduce(|acc, e| acc * e).unwrap_or(0)
    }
}
