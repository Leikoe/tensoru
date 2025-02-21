use std::ops::Add;

use super::view::View;

/// A tracked shape
///
/// This struct allows for expressing a series of transformations.
/// The series starts from the first and ends at the last [`View`].
struct TrackedShape {
    transformations: Vec<View>,
}

impl TrackedShape {
    /// Create a new TrackedShape from a view
    pub fn new(initial_view: View) -> Self {
        TrackedShape {
            transformations: vec![initial_view],
        }
    }

    /// Internal helper to get a reference to the last transformation ([`View`]).
    ///
    /// By constructor invariant, a [`TrackedShape`] always contains atleast one transformation ([`View`]).
    fn last(&self) -> &View {
        self.transformations
            .last()
            .expect("a TrackedShape should always contain atleast 1 View")
    }

    /// Get the shape final shape
    pub fn shape(&self) -> &[usize] {
        self.last().shape()
    }

    /// Get the final strides
    pub fn strides(&self) -> &[usize] {
        self.last().strides()
    }

    /// Get the final size
    pub fn size(&self) -> usize {
        self.last().size()
    }
}

impl<'a> Add<&'a TrackedShape> for &'a TrackedShape {
    type Output = TrackedShape;

    /// Create a new TrackedShape which Tracks the composition of the first and second (each as transformations).
    fn add(self, rhs: Self) -> Self::Output {
        let mut transformations = self.transformations.clone();
        for v in &rhs.transformations {
            transformations.push(v.clone()); // TODO: simplify
        }
        TrackedShape { transformations }
    }
}

#[cfg(test)]
mod test {
    use crate::shape::view::View;

    use super::TrackedShape;

    #[test]
    pub fn shape() {
        let initial_view = View::new(&[2, 2], &[1, 1]);
        let ts = TrackedShape::new(initial_view.clone());

        assert_eq!(initial_view.shape(), ts.shape());
    }
}
