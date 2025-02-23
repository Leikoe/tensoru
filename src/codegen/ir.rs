pub type RegId = u32;

#[derive(Clone, Eq, PartialEq, Debug)]
pub enum Type {
    F16,
    F32,
    F64,
    U32,
    I32,
    Ptr(Box<Type>),
}

pub enum Expr {
    Load(RegId),
    Const(f32), // TODO: find a way to have ints ?
    Deref(Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
}

pub struct Declaration {
    pub reg: RegId,
    pub ty: Type,
    pub is_const: bool,
}

pub enum LValue {
    Reg(RegId),
    Deref(Box<Expr>),
}

pub enum Instruction {
    Define(RegId, Expr),
    Affect(LValue, Expr),
}

pub struct Block(pub Vec<Instruction>);

pub struct Kernel {
    pub name: String,
    pub args: Vec<Declaration>,
    pub body: Block,
}

#[cfg(test)]
mod tests {
    use super::{Declaration, Expr, Instruction, Kernel, Type, Block, LValue};

    #[test]
    fn simple_add_one() {
        // Buff(1) = Buff(0) + 1
        let name = "add".to_string();
        let in_buff = Declaration {
            reg: 0,
            ty: Type::Ptr(Box::new(Type::F16)),
            is_const: true,
        };
        let out_buff = Declaration {
            reg: 1,
            ty: Type::Ptr(Box::new(Type::F16)),
            is_const: false,
        };
        let body = Block(vec![
            Instruction::Define(2, Expr::Deref(Box::new(Expr::Load(in_buff.reg)))),
            Instruction::Define(3, Expr::Add(Box::new(Expr::Load(2)), Box::new(Expr::Const(1.0)))),
            Instruction::Affect(LValue::Deref(Box::new(Expr::Load(1))), Expr::Load(3))
        ]);

        let args = vec![in_buff, out_buff];
        let kernel = Kernel {
            name,
            args,
            body
        };
    }
}
