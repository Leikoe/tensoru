use std::{ops::Range, rc::Rc};

pub type RegId = usize;

#[derive(Debug, Clone, Copy)]
pub enum ScalarType {
    F16,
    F32,
    F64,
    U32,
    I32,
}

#[derive(Debug, Clone, Copy)]
pub enum Type {
    Scalar(ScalarType),
    Vectorized(ScalarType, usize),
}

#[derive(Debug, Clone)]
pub enum Expr {
    Load(Declaration),
    Immediate(f32), // TODO: find a way to have ints ?
    Index(Declaration, Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
}

#[derive(Clone, Copy, Debug)]
pub struct Declaration {
    pub reg: RegId,
    pub ty: Type,
    pub is_const: bool,
}

#[derive(Debug, Clone)]
pub enum LValue {
    Reg(Declaration),
    Index(Declaration, Expr),
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Range(Declaration, Range<usize>, Block),
    Define(Declaration, Expr),
    Affect(LValue, Expr),
}

#[derive(Debug, Clone)]
pub struct Block(Vec<Instruction>);

#[derive(Debug, Clone)]
pub struct KernelBuilder {
    pub name: String,
    pub args: Vec<Declaration>,
    pub body: Block,
    reg_count: usize,
}

impl KernelBuilder {
    fn new<N: ToString>(name: N) -> Self {
        Self {
            name: name.to_string(),
            args: vec![],
            body: Block(vec![]),
            reg_count: 0,
        }
    }

    fn add_input(&mut self, ty: Type) -> Declaration {
        let decl = Declaration {
            reg: self.args.len(),
            ty,
            is_const: true,
        };
        self.args.push(decl);
        self.reg_count += 1;
        decl
    }

    fn add_output(&mut self, ty: Type) -> Declaration {
        let decl = Declaration {
            reg: self.args.len(),
            ty,
            is_const: false,
        };
        self.args.push(decl);
        self.reg_count += 1;
        decl
    }

    fn new_range(&mut self) -> Declaration {
        let decl = Declaration {
            reg: self.reg_count,
            ty: Type::Scalar(ScalarType::U32),
            is_const: false,
        };
        self.reg_count += 1;
        decl
    }
    fn commit_range(&mut self, index: Declaration, range: Range<usize>, block: Block) {
        self.body.0.push(Instruction::Range(index, range, block));
    }

    fn new_reg(&mut self) -> usize {
        let rid = self.reg_count;
        self.reg_count += 1;
        rid
    }

    fn new_def(&mut self, ty: Type, expr: Expr) -> (Declaration, Instruction) {
        let decl = Declaration {
            reg: self.new_reg(),
            ty,
            is_const: false,
        };

        (decl.clone(), Instruction::Define(decl, expr))
    }
}

#[cfg(test)]
mod tests {
    use super::{Block, Expr, Instruction, KernelBuilder, LValue, ScalarType, Type};

    #[test]
    fn simple_vectoradd() {
        let mut kb = KernelBuilder::new("vectoradd");
        const N: usize = 1024;
        const DT: ScalarType = ScalarType::F16;
        let a = kb.add_input(Type::Vectorized(DT, N));
        let b = kb.add_input(Type::Vectorized(DT, N));
        let c = kb.add_output(Type::Vectorized(DT, N));

        let i = kb.new_range();
        let (va, va_inst) = kb.new_def(Type::Scalar(DT), Expr::Index(a, Box::new(Expr::Load(i))));
        let (vb, vb_inst) = kb.new_def(Type::Scalar(DT), Expr::Index(b, Box::new(Expr::Load(i))));
        let (vc, vc_inst) = kb.new_def(
            Type::Scalar(DT),
            Expr::Add(Box::new(Expr::Load(va)), Box::new(Expr::Load(vb))),
        );
        let store = Instruction::Affect(LValue::Index(c, Expr::Load(i)), Expr::Load(vc));
        let loop_body = vec![va_inst, vb_inst, vc_inst, store];
        kb.commit_range(i, 0..N, Block(loop_body));

        dbg!(kb);
    }
}
