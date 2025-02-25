use crate::codegen::ir::{Declaration, Expr, Instruction, Kernel, LValue, RegId, ScalarType, Type};
use crate::codegen::render::CStyleRenderer;
use std::fmt::Write;

struct CRender {
    src: String,
    ident: usize,
}

const fn scalar_type_to_ctype(sty: ScalarType) -> &'static str {
    match sty {
        ScalarType::F16 => "half",
        ScalarType::F32 => "float",
        ScalarType::F64 => "double",
        ScalarType::U32 => "unsigned int",
        ScalarType::I32 => "int",
    }
}

fn type_to_ctype(ty: Type) -> String {
    match ty {
        Type::Scalar(sty) => scalar_type_to_ctype(sty).to_owned(),
        Type::Vectorized(sty, _size) => format!("*{}", scalar_type_to_ctype(sty)),
    }
}

impl CRender {
    fn append_rendered_args(&mut self, args: &Vec<Declaration>) {
        let args: Vec<String> = args.iter().map(Self::render_decl).collect();
        write!(self.src, "{}", args.join(", ")).unwrap();
    }

    fn render_register(rid: RegId) -> String {
        format!("r{}", rid)
    }

    fn render_expr(expr: &Expr) -> String {
        match expr {
            Expr::Load(decl) => Self::render_register(decl.reg),
            Expr::Immediate(v) => v.to_string(),
            Expr::Index(arr, index) => format!(
                "*({} + {})",
                Self::render_register(arr.reg),
                Self::render_expr(index)
            ),
            Expr::Add(a, b) => format!("({} + {})", Self::render_expr(a), Self::render_expr(b)),
            Expr::Sub(a, b) => format!("({} - {})", Self::render_expr(a), Self::render_expr(b)),
            Expr::Mul(a, b) => format!("({} * {})", Self::render_expr(a), Self::render_expr(b)),
            Expr::Div(a, b) => format!("({} / {})", Self::render_expr(a), Self::render_expr(b)),
        }
    }

    fn render_lvalue(lv: &LValue) -> String {
        match lv {
            LValue::Reg(decl) => Self::render_register(decl.reg),
            LValue::Index(arr, index) => format!(
                "*({} + {})",
                Self::render_register(arr.reg),
                Self::render_expr(index)
            ),
        }
    }

    fn render_decl(decl: &Declaration) -> String {
        if decl.is_const {
            format!(
                "const {} {}",
                type_to_ctype(decl.ty.clone()),
                Self::render_register(decl.reg)
            )
        } else {
            format!(
                "{} {}",
                type_to_ctype(decl.ty.clone()),
                Self::render_register(decl.reg)
            )
        }
    }

    fn append_rendered_instruction(&mut self, instruction: &Instruction) {
        match instruction {
            Instruction::Range(decl, range, block) => {
                write!(
                    self.src,
                    "for ({} = {}; {} < {}; {}++) ",
                    Self::render_decl(decl),
                    range.start,
                    Self::render_register(decl.reg),
                    range.end,
                    Self::render_register(decl.reg)
                )
                .unwrap();
                self.append_rendered_block(block);
            }
            Instruction::Define(decl, expr) => {
                writeln!(
                    self.src,
                    "{} = {};",
                    Self::render_decl(decl),
                    Self::render_expr(expr)
                )
                .unwrap();
            }
            Instruction::Affect(lv, expr) => {
                writeln!(
                    self.src,
                    "{} = {};",
                    Self::render_lvalue(lv),
                    Self::render_expr(expr)
                )
                .unwrap();
            }
        }
    }

    fn append_rendered_indent(&mut self) {
        for _ in 0..self.ident {
            write!(self.src, "\t").unwrap();
        }
    }
}

impl CStyleRenderer for CRender {
    fn new() -> Self {
        CRender {
            src: String::new(),
            ident: 0,
        }
    }

    fn src(&self) -> &str {
        &self.src
    }

    fn append_rendered_signature(&mut self, kernel: &Kernel) {
        write!(self.src, "void {}(", kernel.name).unwrap();
        self.append_rendered_args(&kernel.args);
        write!(self.src, ") ").unwrap();
    }

    fn append_rendered_block(&mut self, block: &crate::codegen::ir::BlockBuilder) {
        writeln!(self.src, "{{").unwrap();
        self.ident += 1;
        for inst in &block.insts {
            self.append_rendered_indent();
            self.append_rendered_instruction(inst);
        }
        self.ident -= 1;
        self.append_rendered_indent();
        writeln!(self.src, "}}").unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::CRender;
    use crate::backends::Render;
    use crate::codegen::ir::*;

    #[test]
    fn empty_kernel() {
        let mut kb = KernelBuilder::new("empty");
        let body = kb.new_block();
        let k = kb.finalize(body);
        println!("{}", CRender::render(&k));
    }

    #[test]
    fn empty_body_kernel() {
        let mut kb = KernelBuilder::new("empty");
        kb.add_input(Type::Vectorized(ScalarType::F16, 1024));
        let body = kb.new_block();
        let k = kb.finalize(body);
        println!("{}", CRender::render(&k));
    }

    #[test]
    fn vectoradd() {
        let k = crate::codegen::ir::samples::gen_vectoradd();
        println!("{}", CRender::render(&k));
    }

    #[test]
    fn c_render_reduce() {
        let k = crate::codegen::ir::samples::gen_simple_reduce(1024);
        println!("{}", CRender::render(&k))
    }
}
