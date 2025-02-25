pub mod cstyle {
    use crate::backends::Render;
    use crate::codegen::ir::*;
    const fn scalar_type_to_ctype(sty: ScalarType) -> &'static str {
        match sty {
            ScalarType::F16 => "half",
            ScalarType::F32 => "float",
            ScalarType::F64 => "double",
            ScalarType::U32 => "unsigned int",
            ScalarType::I32 => "int",
        }
    }

    pub fn type_to_ctype(ty: Type) -> String {
        match ty {
            Type::Scalar(sty) => scalar_type_to_ctype(sty).to_owned(),
            Type::Vectorized(sty, _size) => format!("*{}", scalar_type_to_ctype(sty)),
        }
    }

    pub trait CStyleRenderer {
        fn new() -> Self;
        fn src(&self) -> &str;
        fn append_rendered_signature(&mut self, kernel: &Kernel);
        fn append_rendered_block(&mut self, block: &BlockBuilder);
    }

    impl<T: CStyleRenderer> Render for T {
        fn render(kernel: &Kernel) -> String {
            let mut render = T::new();
            render.append_rendered_signature(kernel);
            render.append_rendered_block(&kernel.body);
            render.src().to_string()
        }
    }
}
