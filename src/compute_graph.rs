use crate::dtype::DType;

enum Op {
    Add(),
    Sub,
    Mul,
    Div,
}

enum Value {
    Op(Op),
    Const(),
}
