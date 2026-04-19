#[derive(Debug, Clone)]
pub enum Expr {
    Number(f64),
    Boolean(bool),
    StringLiteral(String),
    Identifier(String),
    UnaryOp { op: String, right: Box<Expr> },
    BinaryOp { left: Box<Expr>, op: String, right: Box<Expr> },
    MatrixOp { left: Box<Expr>, op: String, right: Box<Expr> },
    FunctionCall { name: String, args: Vec<Expr> },
    Array(Vec<Expr>),
    Dictionary(Vec<(Expr, Expr)>),
    SliceAccess { target: Box<Expr>, start: Option<Box<Expr>>, end: Option<Box<Expr>> },
    IndexAccess { target: Box<Expr>, index: Box<Expr> },
    MultiIndexAccess { target: Box<Expr>, indices: Vec<Expr> },
}

#[derive(Debug, Clone)]
pub enum Statement {
    Expression(Expr),
    Assignment { name: String, value: Expr },
    IndexAssignment { target: Expr, index: Expr, value: Expr },
    MultiIndexAssignment { target: Expr, indices: Vec<Expr>, value: Expr },
    Print { value: Expr },
    If { condition: Expr, then_branch: Vec<Statement>, else_branch: Option<Vec<Statement>> },
    While { condition: Expr, body: Vec<Statement> },
    FunctionDecl { name: String, params: Vec<String>, body: Vec<Statement> },
    Return { value: Expr },
    Break,
    Continue,
    GpuBlock { statements: Vec<Statement> },
    Import { module: String },
    TryCatch { try_block: Vec<Statement>, error_var: String, catch_block: Vec<Statement> },
    Throw { value: Expr },
    For { iterator: String, start: Expr, end: Expr, body: Vec<Statement> },
    ForEach { iterator: String, array: Expr, body: Vec<Statement> },
}