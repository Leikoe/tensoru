use crate::backends::Renderer;
use crate::codegen::ir::{Declaration, Expr, Instruction, Kernel, LValue, RegId, Type};
use std::fmt::{format, Write};
use std::ptr::write;

struct CRenderer {
}

fn type_to_ctype(ty: Type) -> String {
    match ty {
        Type::F16 => "half".to_string(),
        Type::F32 => "float".to_string(),
        Type::F64 => "double".to_string(),
        Type::U32 => "unsigned int".to_string(),
        Type::I32 => "int".to_string(),
        Type::Ptr(t) => format!("*{}", type_to_ctype(*t)),
    }
}

impl CRenderer {
    fn render_register(rid: RegId) -> String {
        format!("r{}", rid)
    }

    fn render_expr(expr: &Expr) -> String {
        match expr {
            Expr::Load(rid) => Self::render_register(*rid),
            Expr::Const(v) => v.to_string(),
            Expr::Deref(v) => format!("*{}", Self::render_expr(v)),
            Expr::Add(a, b) => format!("({} + {})", Self::render_expr(a), Self::render_expr(b)),
            Expr::Sub(a, b) => format!("({} - {})", Self::render_expr(a), Self::render_expr(b)),
            Expr::Mul(a, b) => format!("({} * {})", Self::render_expr(a), Self::render_expr(b)),
            Expr::Div(a, b) => format!("({} / {})", Self::render_expr(a), Self::render_expr(b)),
        }
    }

    fn render_lvalue(lv: &LValue) -> String {
        match lv {
            LValue::Reg(rid) => Self::render_register(*rid),
            LValue::Deref(v) => format!("*{}", Self::render_expr(v)),
        }
    }

    fn render_decl(decl: &Declaration) -> String {
        if decl.is_const {
            format!("const {}: {}", Self::render_register(decl.reg), type_to_ctype(decl.ty.clone()))
        } else {
            format!("{}: {}", Self::render_register(decl.reg), type_to_ctype(decl.ty.clone()))
        }
    }

    fn append_rendered_instruction(f: &mut String, instruction: &Instruction) {
        match instruction {
            Instruction::Define(rid, expr) => {
                write!(f, "{} = {}", Self::render_register(*rid), Self::render_expr(expr)).unwrap();
            }
            Instruction::Affect(lv, expr) => {
                write!(f, "{} = {}", Self::render_lvalue(lv), Self::render_expr(expr)).unwrap();
            }
        }
        write!(f, ";\n").unwrap();
    }
}

impl Renderer for CRenderer {
    fn render(kernel: &Kernel) -> String {
        let mut src = String::new();
        write!(src, "void {}(", kernel.name).unwrap();
        let args: Vec<String> = kernel.args.iter().map(Self::render_decl).collect();
        write!(src, "{}", args.join(", ")).unwrap();
        writeln!(src, ") {{").unwrap();
        for inst in &kernel.body.0 {
            write!(src, "\t").unwrap();
            Self::append_rendered_instruction(&mut src, inst);
        }
        writeln!(src, "}}").unwrap();
        src
    }
}

#[cfg(test)]
mod tests {
    use crate::codegen::ir::{Block, Declaration, Expr, Instruction, Kernel, LValue, Type};
    use crate::codegen::render::CRenderer;
    use crate::backends::Renderer;

    #[test]
    fn empty_kernel() {
        let k = Kernel {
            name: "empty".to_string(),
            args: vec![],
            body: Block(vec![])
        };
        dbg!(CRenderer::render(&k));
    }

    #[test]
    fn empty_body_kernel() {
        let k = Kernel {
            name: "empty".to_string(),
            args: vec![Declaration {
                reg: 0,
                ty: Type::Ptr(Box::new(Type::F16)),
                is_const: false,
            }],
            body: Block(vec![])
        };
        dbg!(CRenderer::render(&k));
    }

    #[test]
    fn dummy_kernel() {
        let k = Kernel {
            name: "empty".to_string(),
            args: vec![
                Declaration {
                    reg: 0,
                    ty: Type::Ptr(Box::new(Type::F16)),
                    is_const: true,
                },
                Declaration {
                   reg: 1,
                   ty: Type::Ptr(Box::new(Type::F16)),
                   is_const: false,
                }],
            body: Block(vec![
                Instruction::Define(2, Expr::Deref(Box::new(Expr::Load(0)))),
                Instruction::Define(3, Expr::Add(Box::new(Expr::Load(1)), Box::new(Expr::Const(1.)))),
                Instruction::Affect(LValue::Deref(Box::new(Expr::Load(1))), Expr::Load(3))
            ])
        };
        println!("{}", CRenderer::render(&k));
    }
}