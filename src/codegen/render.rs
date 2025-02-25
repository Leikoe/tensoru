use crate::backends::Render;
use crate::codegen::ir::{BlockBuilder, Kernel};

pub trait CStyleRenderer {
    fn new() -> Self;
    fn src(&self) -> &str;
    fn append_rendered_signature(&mut self, kernel: &Kernel);
    fn append_rendered_block(&mut self, block: &BlockBuilder);
}

impl<T: CStyleRenderer> Render for T {
    fn render(kernel: &super::ir::Kernel) -> String {
        let mut render = T::new();
        render.append_rendered_signature(kernel);
        render.append_rendered_block(&kernel.body);
        render.src().to_string()
    }
}
