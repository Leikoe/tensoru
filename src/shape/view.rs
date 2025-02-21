// use core::range::Range;

#[derive(Clone, Debug)]
pub struct View {
    shape: Vec<usize>,
    strides: Vec<usize>,
    contiguous: bool,
}

/// construct strides vector from a shape (assumes contiguous)
fn strides_from_shape(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut stride = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = stride;
        stride *= shape[i];
    }
    fix_1_sized_dims_strides(shape, strides)
}

/// replace strides for len 1 dims by 0
fn fix_1_sized_dims_strides(shape: &[usize], strides: Vec<usize>) -> Vec<usize> {
    shape
        .into_iter()
        .zip(strides.into_iter())
        .map(|(&s, st)| match s {
            1 => 0,
            _ => st,
        })
        .collect()
}

impl View {
    pub fn new(shape: &[usize], strides: &[usize]) -> Self {
        assert_eq!(shape.len(), strides.len(), "Shape and strides must match");
        let contiguous = strides == strides_from_shape(shape);

        Self {
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            contiguous,
        }
    }

    pub fn from_shape(shape: &[usize]) -> Self {
        let strides = strides_from_shape(shape);
        Self {
            shape: shape.to_vec(),
            strides,
            contiguous: true,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_contiguous(&self) -> bool {
        self.contiguous
    }

    pub fn indexing_expr(&self, buffer: &str, indices: &[&str]) -> String {
        assert_eq!(
            self.shape.len(),
            indices.len(),
            "Number of indices must match shape"
        );

        let mut offset = "0".to_string();
        for i in 0..self.shape.len() {
            let term = format!("{} * {}", indices[i].to_string(), self.strides[i]);
            offset = format!("{offset} + {term}");
        }

        format!("{buffer}[{offset}]")
    }

    pub fn permute_axes(&self, permutation: &[usize]) -> Self {
        let permuted_axes_sorted: Vec<usize> = {
            let mut permuted_axes = permutation.to_vec();
            permuted_axes.sort();
            permuted_axes
        };
        let n_axes = self.shape.len();
        assert!(
            permuted_axes_sorted.into_iter().eq(0..n_axes),
            "a permutation should contain exactly all the axes ({:?})",
            0..n_axes
        );

        let shape: Vec<usize> = permutation.iter().map(|i| self.shape[*i]).collect();
        let strides: Vec<usize> = permutation.iter().map(|i| self.strides[*i]).collect();
        View::new(&shape, &strides) // TODO: reused the allocated shape & strides
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Option<Self> {
        if self.shape() == new_shape {
            return Some(self.clone());
        }

        if self.is_contiguous() {
            return Some(View::from_shape(new_shape));
        }

        None
    }
}

#[cfg(test)]
mod test {
    use super::View;
    use crate::shape::view::{fix_1_sized_dims_strides, strides_from_shape};

    #[test]
    fn test_strides_from_shape_simple() {
        let shape = [2, 2];
        assert_eq!(vec![2, 1], strides_from_shape(&shape));
    }

    #[test]
    fn test_1_sized_dim_stride() {
        assert_eq!(
            vec![2, 0, 1],
            fix_1_sized_dims_strides(&[2, 1, 2], vec![2, 2, 1])
        );
    }

    #[test]
    fn size_simple() {
        let v = View::new(&[2, 2], &[2, 1]);
        assert_eq!(4, v.size(), "mismatched view size");
        assert!(v.is_contiguous());
    }

    #[test]
    fn test_indexing_expr() {
        let v = View::from_shape(&[32, 16, 8]);
        println!("{}", v.indexing_expr("buff", &["i", "j", "k"]));
    }

    // PERMUTES

    #[test]
    fn test_permute_axes_simple() {
        let v = View::from_shape(&[32, 16]);
        let reversed_strides: Vec<usize> = v.strides().iter().rev().copied().collect();
        let pv = v.permute_axes(&[1, 0]);
        assert_eq!([16, 32], pv.shape());
        assert_eq!(&reversed_strides, pv.strides());

        dbg!(pv.indexing_expr("buff", &["i", "j"]));
    }

    #[test]
    fn test_permute_axes_two_last() {
        let v = View::from_shape(&[64, 32, 16]);
        let pv = v.permute_axes(&[0, 2, 1]);
        assert_eq!([64, 16, 32], pv.shape());
        assert_eq!([512, 1, 16], pv.strides());
    }

    #[test]
    #[should_panic]
    fn test_permute_axes_missing_axis() {
        let v = View::from_shape(&[32, 16]);
        v.permute_axes(&[1]);
    }

    // RESHAPES

    #[test]
    fn test_reshape_simple() {
        let v = View::from_shape(&[3, 2]);
        assert!(v.reshape(&[2, 3]).is_some());
    }

    #[test]
    fn test_reshape_simple_after_permute() {
        let v = View::from_shape(&[3, 2]);
        let pv = v.permute_axes(&[1, 0]);
        assert!(pv.reshape(&[3, 2]).is_none());
    }
}
