use std::{
    ops::{Index, Range},
    rc::Rc,
    slice::SliceIndex,
    sync::{Arc, Mutex},
};

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

impl Declaration {
    pub fn index(&self, index: Expr) -> Expr {
        Expr::Index(*self, Box::new(index))
    }
}

#[derive(Debug, Clone)]
pub enum LValue {
    Reg(Declaration),
    Index(Declaration, Expr),
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Range(Declaration, Range<usize>, BlockBuilder),
    Define(Declaration, Expr),
    Affect(LValue, Expr),
}

#[derive(Debug, Clone)]
pub struct BlockBuilder {
    pub insts: Vec<Instruction>,
    kernel_handle: Arc<Mutex<InnerKernelBuilder>>,
}

impl BlockBuilder {
    pub fn new(kernel_handle: Arc<Mutex<InnerKernelBuilder>>) -> Self {
        Self {
            insts: vec![],
            kernel_handle,
        }
    }

    pub fn define(&mut self, ty: Type, expr: Expr) -> Declaration {
        let decl = Declaration {
            reg: self.new_reg(),
            ty,
            is_const: false,
        };

        self.insts.push(Instruction::Define(decl, expr));
        decl
    }

    pub fn affect(&mut self, lvalue: LValue, expr: Expr) {
        self.insts.push(Instruction::Affect(lvalue, expr));
    }

    fn new_reg(&mut self) -> usize {
        self.kernel_handle.lock().unwrap().new_reg()
    }

    pub fn new_range(&mut self, range: Range<usize>) -> (Declaration, &mut BlockBuilder) {
        let decl = Declaration {
            reg: self.new_reg(),
            ty: Type::Scalar(ScalarType::U32),
            is_const: false,
        };
        self.insts.push(Instruction::Range(
            decl,
            range,
            BlockBuilder::new(self.kernel_handle.clone()),
        ));
        if let Instruction::Range(_, _, b) = self.insts.last_mut().expect("we just pushed") {
            (decl, b)
        } else {
            unreachable!()
        }
    }
}

#[derive(Debug, Clone)]
pub struct KernelBuilder(Arc<Mutex<InnerKernelBuilder>>);

#[derive(Debug, Clone)]
pub struct InnerKernelBuilder {
    name: String,
    args: Vec<Declaration>,
    reg_count: usize,
}

impl InnerKernelBuilder {
    fn new_reg(&mut self) -> usize {
        let rid = self.reg_count;
        self.reg_count += 1;
        rid
    }
}

#[derive(Debug, Clone)]
pub struct Kernel {
    pub name: String,
    pub args: Vec<Declaration>,
    pub body: BlockBuilder,
}

impl KernelBuilder {
    pub fn new<N: ToString>(name: N) -> Self {
        Self(Arc::new(Mutex::new(InnerKernelBuilder {
            name: name.to_string(),
            args: vec![],
            reg_count: 0,
        })))
    }

    pub fn add_input(&mut self, ty: Type) -> Declaration {
        let decl = Declaration {
            reg: self.0.lock().unwrap().args.len(),
            ty,
            is_const: true,
        };
        self.0.lock().unwrap().args.push(decl);
        self.0.lock().unwrap().reg_count += 1;
        decl
    }

    pub fn add_output(&mut self, ty: Type) -> Declaration {
        let decl = Declaration {
            reg: self.0.lock().unwrap().args.len(),
            ty,
            is_const: false,
        };
        self.0.lock().unwrap().args.push(decl);
        self.0.lock().unwrap().reg_count += 1;
        decl
    }

    pub fn new_block(&mut self) -> BlockBuilder {
        BlockBuilder::new(self.0.clone())
    }

    pub fn finalize(self, body: BlockBuilder) -> Kernel {
        let name = self.0.lock().unwrap().name.clone();
        let args = self.0.lock().unwrap().args.clone();
        Kernel { name, args, body }
    }
}

pub mod samples {
    use super::*;

    pub fn gen_simple_reduce(n: usize) -> Kernel {
        let mut kb = KernelBuilder::new("reduce");
        const ELEM_TYPE: ScalarType = ScalarType::F16;
        let a = kb.add_input(Type::Vectorized(ELEM_TYPE, n));
        let o = kb.add_output(Type::Vectorized(ELEM_TYPE, 1));

        let mut body = kb.new_block();
        let acc = body.define(Type::Scalar(ELEM_TYPE), Expr::Immediate(0.));
        let (i, lb) = body.new_range(0..n);
        let v = lb.define(Type::Scalar(ELEM_TYPE), a.index(Expr::Load(i)));
        lb.affect(
            LValue::Reg(acc),
            Expr::Add(Box::new(Expr::Load(acc)), Box::new(Expr::Load(v))),
        );
        body.affect(LValue::Index(o, Expr::Immediate(0.)), Expr::Load(acc));

        kb.finalize(body)
    }

    pub fn gen_vectoradd() -> Kernel {
        let mut kb = KernelBuilder::new("vectoradd");
        const N: usize = 1024;
        const DT: ScalarType = ScalarType::F16;
        let a = kb.add_input(Type::Vectorized(DT, N));
        let b = kb.add_input(Type::Vectorized(DT, N));
        let c = kb.add_output(Type::Vectorized(DT, N));

        let mut body = kb.new_block();

        let (i, rblock) = body.new_range(0..N);
        let va = rblock.define(Type::Scalar(DT), a.index(Expr::Load(i)));
        let vb = rblock.define(Type::Scalar(DT), b.index(Expr::Load(i)));
        let vc = rblock.define(
            Type::Scalar(DT),
            Expr::Add(Box::new(Expr::Load(va)), Box::new(Expr::Load(vb))),
        );
        rblock.affect(LValue::Index(c, Expr::Load(i)), Expr::Load(vc));

        kb.finalize(body)
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn simple_reduce() {
        let k = samples::gen_simple_reduce(1024);
        dbg!(k);
    }

    #[test]
    fn simple_vectoradd() {
        let k = samples::gen_vectoradd();
        dbg!(k);
    }
}
