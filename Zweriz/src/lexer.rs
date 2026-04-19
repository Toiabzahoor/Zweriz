use logos::Logos;

#[derive(Logos, Debug, PartialEq, Clone)]
#[logos(skip r"[ \t\n\f\r]+")]
#[logos(skip r"#.*")]
pub enum Token {
    #[token("try")] TryKeyword,
    #[token("catch")] CatchKeyword,
    #[token("throw")] ThrowKeyword,
    #[token("import")] ImportKeyword,
    #[token("class")] ClassKeyword,
    #[token("fn")] FnKeyword,
    #[token("for")] ForKeyword,
    #[token("in")] InKeyword,
    #[token("return")] ReturnKeyword,
    #[token("break")] BreakKeyword,
    #[token("continue")] ContinueKeyword,
    #[token("print")] PrintKeyword,
    #[token("if")] IfKeyword,
    #[token("else")] ElseKeyword,
    #[token("while")] WhileKeyword,
    #[token("GPU")] GpuKeyword,
    #[token("true")] TrueKeyword,
    #[token("false")] FalseKeyword,
    #[token("not")] NotKeyword,
    #[token("and")] AndKeyword,
    #[token("or")] OrKeyword,

    #[token("==")] EqEq,
    #[token("!=")] NotEq,
    #[token(">=")] GtEq,
    #[token("<=")] LtEq,
    #[token("<")] Lt,
    #[token(">")] Gt,
    #[token("&&")] AndAnd,
    #[token("||")] OrOr,

    #[token("+=")] PlusAssign,
    #[token("-=")] MinusAssign,
    #[token("*=")] StarAssign,
    #[token("/=")] SlashAssign,
    #[token("%=")] ModuloAssign,
    #[token("**=")] PowerAssign,
    #[token("=")] Assign,

    #[token("+")] Plus,
    #[token("-")] Minus,
    #[token("**")] Power,
    #[token("*")] Star,
    #[token("/")] Slash,
    #[token("%")] Modulo,
    #[token("|")] Pipe,
    #[token("&")] Ampersand,
    #[token("^")] Caret,
    #[token("<<")] Shl,
    #[token(">>")] Shr,
    #[token("~")] Tilde,
    #[token("@")] At,

    #[token("(")] LParen,
    #[token(")")] RParen,
    #[token("[")] LBracket,
    #[token("]")] RBracket,
    #[token("{")] LBrace,
    #[token("}")] RBrace,
    #[token(",")] Comma,
    #[token(":")] Colon,
    #[token(".")] Dot,
    #[token(";")] Semicolon,

    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string())]
    Ident(String),

    #[regex(r"[0-9]+(\.[0-9]+)?", |lex| lex.slice().parse::<f64>().unwrap())]
    Number(f64),

    #[regex(r#""(?:[^"\\]|\\.)*""#, |lex| {
        let s = lex.slice();
        s[1..s.len()-1].to_string()
    })]
    StringLiteral(String),

    #[regex(r#"f"(?:[^"\\]|\\.)*""#, |lex| {
        let s = lex.slice();
        s[2..s.len()-1].to_string()
    })]
    FString(String),
}