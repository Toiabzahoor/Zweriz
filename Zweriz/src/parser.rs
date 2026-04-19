use logos::{Logos, Span};
use crate::lexer::Token;
use crate::ast::{Expr, Statement};

pub struct Parser<'a> {
    lexer: logos::Lexer<'a, Token>, current_token: Option<Token>,
    current_span: Span, previous_span: Span, source: &'a str,
}

impl<'a> Parser<'a> {
    pub fn new(source: &'a str) -> Self {
        let mut lexer = Token::lexer(source); let current_token = match lexer.next() { Some(Ok(tok)) => Some(tok), _ => None, };
        let current_span = lexer.span();
        Self { lexer, current_token, current_span: current_span.clone(), previous_span: current_span, source }
    }

    fn error(&self, message: &str, span: Span) -> String {
        let mut line_num = 1; let mut line_start = 0;
        for (i, c) in self.source.char_indices() { if i >= span.start { break; } if c == '\n' { line_num += 1; line_start = i + 1; } }
        let line_end = self.source[line_start..].find('\n').map(|i| line_start + i).unwrap_or(self.source.len());
        let col = span.start.saturating_sub(line_start); let span_len = span.end.saturating_sub(span.start).max(1);
        format!("--> Line {}, Column {}\n   |\n{:<2} | {}\n   | {}{}\n   | {}", line_num, col + 1, line_num, &self.source[line_start..line_end], " ".repeat(col), "^".repeat(span_len), message)
    }

    fn peek(&self) -> Option<&Token> { self.current_token.as_ref() }

    fn advance(&mut self) -> Result<Option<Token>, String> {
        let token = self.current_token.take(); self.previous_span = self.current_span.clone();
        if let Some(res) = self.lexer.next() { match res { Ok(tok) => self.current_token = Some(tok), Err(_) => return Err(self.error("Unrecognized character.", self.lexer.span())), } self.current_span = self.lexer.span(); }
        Ok(token)
    }

    fn expect(&mut self, expected: Token) -> Result<(), String> {
        let span = self.current_span.clone(); let token = self.advance()?;
        if token != Some(expected.clone()) { let found = token.map(|t| format!("{:?}", t)).unwrap_or_else(|| "EOF".to_string()); return Err(self.error(&format!("Expected {:?}, but found {}", expected, found), span)); } Ok(())
    }

    pub fn parse(&mut self) -> Result<Vec<Statement>, String> {
        let mut statements = Vec::new();
        while self.peek().is_some() {
            if self.peek() == Some(&Token::Semicolon) { self.advance()?; continue; }
            statements.push(self.parse_statement()?);
            if self.peek() == Some(&Token::Semicolon) { self.advance()?; }
        }
        Ok(statements)
    }

    fn parse_statement(&mut self) -> Result<Statement, String> {
        let token = match self.peek() { Some(t) => t.clone(), None => return Err(self.error("Unexpected EOF", self.current_span.clone())), };
        match token {
            Token::TryKeyword => {
                self.advance()?; self.expect(Token::LBrace)?; let try_block = self.parse_block()?; self.expect(Token::CatchKeyword)?; self.expect(Token::LParen)?;
                let error_var = if let Some(Token::Ident(n)) = self.advance()? { n } else { return Err(self.error("Expected error variable name", self.current_span.clone())); };
                self.expect(Token::RParen)?; self.expect(Token::LBrace)?; let catch_block = self.parse_block()?;
                Ok(Statement::TryCatch { try_block, error_var, catch_block })
            }
            Token::ThrowKeyword => { self.advance()?; Ok(Statement::Throw { value: self.parse_expr()? }) }
            Token::ImportKeyword => { self.advance()?; let span = self.current_span.clone(); let module = if let Some(Token::Ident(n)) = self.advance()? { n } else { return Err(self.error("Expected module name", span)); }; Ok(Statement::Import { module }) }
            Token::ClassKeyword => {
                self.advance()?; let span = self.current_span.clone(); let name = if let Some(Token::Ident(n)) = self.advance()? { n } else { return Err(self.error("Expected class name", span)); };
                self.expect(Token::LBrace)?; let mut props = Vec::new();
                while self.peek() != Some(&Token::RBrace) {
                    if self.peek() == Some(&Token::Semicolon) { self.advance()?; continue; }
                    let p_span = self.current_span.clone(); let prop_name = if let Some(Token::Ident(n)) = self.advance()? { n } else { return Err(self.error("Expected property", p_span)); }; self.expect(Token::Assign)?; props.push((Expr::StringLiteral(prop_name), self.parse_expr()?));
                    if self.peek() == Some(&Token::Semicolon) { self.advance()?; }
                }
                self.expect(Token::RBrace)?; Ok(Statement::FunctionDecl { name, params: vec![], body: vec![ Statement::Return { value: Expr::Dictionary(props) } ] })
            }
            Token::FnKeyword => {
                self.advance()?; let span = self.current_span.clone(); let name = if let Some(Token::Ident(n)) = self.advance()? { n } else { return Err(self.error("Expected function name", span)); };
                self.expect(Token::LParen)?; let mut params = Vec::new();
                if self.peek() != Some(&Token::RParen) { loop { let p_span = self.current_span.clone(); if let Some(Token::Ident(p)) = self.advance()? { params.push(p); } else { return Err(self.error("Expected parameter", p_span)); } if self.peek() == Some(&Token::Comma) { self.advance()?; } else { break; } } }
                self.expect(Token::RParen)?; self.expect(Token::LBrace)?; Ok(Statement::FunctionDecl { name, params, body: self.parse_block()? })
            }
            Token::ForKeyword => {
                self.advance()?; let span = self.current_span.clone();
                let iterator = if let Some(Token::Ident(n)) = self.advance()? { n } else { return Err(self.error("Expected iterator variable name", span)); };
                let next_tok = self.advance()?;

                if next_tok == Some(Token::Assign) {
                    let start = self.parse_expr()?;
                    if let Some(Token::Ident(n)) = self.advance()? {
                        if n == "to" {
                            let end = self.parse_expr()?;
                            self.expect(Token::LBrace)?; let body = self.parse_block()?;
                            return Ok(Statement::For { iterator, start, end, body });
                        }
                    }
                    return Err(self.error("Expected 'to' in standard for loop. E.g., 'for i = 0 to 10'", self.current_span.clone()));
                } else if next_tok == Some(Token::InKeyword) {
                    let array = self.parse_expr()?;
                    self.expect(Token::LBrace)?; let body = self.parse_block()?;
                    return Ok(Statement::ForEach { iterator, array, body });
                }
                Err(self.error("Invalid for loop. Use 'for i = 0 to 10' or 'for x in array'", self.current_span.clone()))
            }
            Token::ReturnKeyword => { self.advance()?; Ok(Statement::Return { value: self.parse_expr()? }) }
            Token::BreakKeyword => { self.advance()?; Ok(Statement::Break) }
            Token::ContinueKeyword => { self.advance()?; Ok(Statement::Continue) }
            Token::PrintKeyword => { self.advance()?; self.expect(Token::LParen)?; let value = self.parse_expr()?; self.expect(Token::RParen)?; Ok(Statement::Print { value }) }
            Token::IfKeyword => {
                self.advance()?; let condition = self.parse_expr()?; self.expect(Token::LBrace)?; let then_branch = self.parse_block()?; let mut else_branch = None;
                if self.peek() == Some(&Token::ElseKeyword) { self.advance()?; self.expect(Token::LBrace)?; else_branch = Some(self.parse_block()?); }
                Ok(Statement::If { condition, then_branch, else_branch })
            }
            Token::WhileKeyword => { self.advance()?; let condition = self.parse_expr()?; self.expect(Token::LBrace)?; Ok(Statement::While { condition, body: self.parse_block()? }) }
            Token::GpuKeyword => {
                self.advance()?; self.expect(Token::LBrace)?; Ok(Statement::GpuBlock { statements: self.parse_block()? })
            }
            _ => {
                let expr = self.parse_expr()?;
                let is_assign = matches!(self.peek(), Some(Token::Assign | Token::PlusAssign | Token::MinusAssign | Token::StarAssign | Token::SlashAssign | Token::ModuloAssign | Token::PowerAssign));

                if is_assign {
                    let op_tok = self.advance()?.unwrap();
                    let mut value = self.parse_expr()?;

                    if op_tok != Token::Assign {
                        let op_str = match op_tok {
                            Token::PlusAssign => "+",
                            Token::MinusAssign => "-",
                            Token::StarAssign => "*",
                            Token::SlashAssign => "/",
                            Token::ModuloAssign => "%",
                            Token::PowerAssign => "**",
                            _ => unreachable!(),
                        };
                        value = Expr::BinaryOp {
                            left: Box::new(expr.clone()),
                            op: op_str.to_string(),
                            right: Box::new(value),
                        };
                    }

                    match expr {
                        Expr::Identifier(name) => Ok(Statement::Assignment { name, value }),
                        Expr::IndexAccess { target, index } => Ok(Statement::IndexAssignment { target: *target, index: *index, value }),
                        Expr::MultiIndexAccess { target, indices } => Ok(Statement::MultiIndexAssignment { target: *target, indices, value }),
                        _ => Err(self.error("Invalid assignment target.", self.current_span.clone())),
                    }
                } else { Ok(Statement::Expression(expr)) }
            }
        }
    }

    fn parse_block(&mut self) -> Result<Vec<Statement>, String> {
        let mut statements = Vec::new();
        while let Some(tok) = self.peek() {
            if tok == &Token::RBrace { break; }
            if tok == &Token::Semicolon { self.advance()?; continue; }
            statements.push(self.parse_statement()?);
            if self.peek() == Some(&Token::Semicolon) { self.advance()?; }
        }
        self.expect(Token::RBrace)?; Ok(statements)
    }

    fn parse_expr(&mut self) -> Result<Expr, String> { self.parse_expr_bp(0) }

    fn infix_binding_power(&self, op: &Token) -> Option<(u8, u8)> {
        match op {
            Token::OrKeyword | Token::Pipe | Token::OrOr => Some((5, 6)),
            Token::AndKeyword | Token::Ampersand | Token::AndAnd => Some((7, 8)),
            Token::Caret => Some((9, 10)),
            Token::EqEq | Token::Lt | Token::Gt | Token::GtEq | Token::LtEq | Token::NotEq => Some((10, 11)),
            Token::Shl | Token::Shr => Some((15, 16)),
            Token::Plus | Token::Minus => Some((20, 21)),
            Token::Star | Token::Slash | Token::Modulo | Token::At => Some((30, 31)),
            Token::Power => Some((41, 40)),
            Token::Ident(name) if name.starts_with("M_") => Some((40, 41)),
            _ => None,
        }
    }

    fn parse_expr_bp(&mut self, min_bp: u8) -> Result<Expr, String> {
        let mut lhs = self.parse_prefix()?;
        loop {
            let op_tok = match self.peek() { Some(tok) => tok.clone(), _ => break };
            if let Some((l_bp, r_bp)) = self.infix_binding_power(&op_tok) {
                if l_bp < min_bp { break; } self.advance()?; let rhs = self.parse_expr_bp(r_bp)?;

                match op_tok {
                    Token::GtEq => { lhs = Expr::UnaryOp { op: "not".to_string(), right: Box::new(Expr::BinaryOp { left: Box::new(lhs), op: "<".to_string(), right: Box::new(rhs) }) }; }
                    Token::LtEq => { lhs = Expr::UnaryOp { op: "not".to_string(), right: Box::new(Expr::BinaryOp { left: Box::new(lhs), op: ">".to_string(), right: Box::new(rhs) }) }; }
                    Token::NotEq => { lhs = Expr::UnaryOp { op: "not".to_string(), right: Box::new(Expr::BinaryOp { left: Box::new(lhs), op: "==".to_string(), right: Box::new(rhs) }) }; }
                    Token::AndAnd => { lhs = Expr::BinaryOp { left: Box::new(lhs), op: "and".to_string(), right: Box::new(rhs) }; }
                    Token::OrOr => { lhs = Expr::BinaryOp { left: Box::new(lhs), op: "or".to_string(), right: Box::new(rhs) }; }
                    _ => {
                        let op_str = match op_tok { Token::Plus => "+", Token::Minus => "-", Token::Star => "*", Token::Slash => "/", Token::Modulo => "%", Token::Power => "**", Token::Caret => "^", Token::Pipe => "|", Token::Ampersand => "&", Token::Shl => "<<", Token::Shr => ">>", Token::EqEq => "==", Token::Lt => "<", Token::Gt => ">", Token::Ident(ref n) => n, Token::AndKeyword => "and", Token::OrKeyword => "or", Token::At => "@", _ => unreachable!(), };
                        if op_str == "@" || op_str.starts_with("M_") { lhs = Expr::MatrixOp { left: Box::new(lhs), op: op_str.to_string(), right: Box::new(rhs) }; } else { lhs = Expr::BinaryOp { left: Box::new(lhs), op: op_str.to_string(), right: Box::new(rhs) }; }
                    }
                }
                continue;
            } break;
        } Ok(lhs)
    }

    fn parse_prefix(&mut self) -> Result<Expr, String> {
        let span = self.current_span.clone(); let token = self.advance()?.ok_or_else(|| self.error("Unexpected EOF", span.clone()))?;
        let mut expr = match token {
            Token::TrueKeyword => Expr::Boolean(true), Token::FalseKeyword => Expr::Boolean(false), Token::Number(val) => Expr::Number(val), Token::StringLiteral(text) => Expr::StringLiteral(text),
            Token::FString(text) => {
                let mut current_expr: Option<Expr> = None; let mut chars = text.chars().peekable(); let mut literal_part = String::new();
                while let Some(c) = chars.next() {
                    if c == '{' {
                        if !literal_part.is_empty() { let processed = literal_part.replace("\\n", "\n").replace("\\t", "\t").replace("\\\"", "\"").replace("\\\\", "\\"); let lit_expr = Expr::StringLiteral(processed); current_expr = match current_expr { Some(e) => Some(Expr::BinaryOp { left: Box::new(e), op: "+".to_string(), right: Box::new(lit_expr) }), None => Some(lit_expr), }; literal_part.clear(); }
                        let mut inner_expr_str = String::new(); while let Some(&next_c) = chars.peek() { if next_c == '}' { chars.next(); break; } inner_expr_str.push(chars.next().unwrap()); }
                        let mut sub_parser = Parser::new(&inner_expr_str); let inner_expr = sub_parser.parse_expr()?;
                        current_expr = match current_expr { Some(e) => Some(Expr::BinaryOp { left: Box::new(e), op: "+".to_string(), right: Box::new(inner_expr) }), None => Some(inner_expr), };
                    } else { literal_part.push(c); }
                }
                if !literal_part.is_empty() { let processed = literal_part.replace("\\n", "\n").replace("\\t", "\t").replace("\\\"", "\"").replace("\\\\", "\\"); let lit_expr = Expr::StringLiteral(processed); current_expr = match current_expr { Some(e) => Some(Expr::BinaryOp { left: Box::new(e), op: "+".to_string(), right: Box::new(lit_expr) }), None => Some(lit_expr), }; }
                current_expr.unwrap_or(Expr::StringLiteral("".to_string()))
            }
            Token::Minus => Expr::UnaryOp { op: "-".to_string(), right: Box::new(self.parse_expr_bp(45)?) }, Token::NotKeyword => Expr::UnaryOp { op: "not".to_string(), right: Box::new(self.parse_expr_bp(15)?) }, Token::Tilde => Expr::UnaryOp { op: "~".to_string(), right: Box::new(self.parse_expr_bp(45)?) }, Token::Ident(name) => Expr::Identifier(name),
            Token::LParen => { let inner = self.parse_expr()?; self.expect(Token::RParen)?; inner }
            Token::LBracket => { let mut elements = Vec::new(); if self.peek() != Some(&Token::RBracket) { loop { elements.push(self.parse_expr()?); if self.peek() == Some(&Token::Comma) { self.advance()?; } else { break; } } } self.expect(Token::RBracket)?; Expr::Array(elements) }
            Token::LBrace => { let mut pairs = Vec::new(); if self.peek() != Some(&Token::RBrace) { loop { let key = self.parse_expr()?; self.expect(Token::Colon)?; let value = self.parse_expr()?; pairs.push((key, value)); if self.peek() == Some(&Token::Comma) { self.advance()?; } else { break; } } } self.expect(Token::RBrace)?; Expr::Dictionary(pairs) }
            _ => return Err(self.error(&format!("Expected expression, found {:?}", token), span)),
        };

        while self.peek() == Some(&Token::LBracket) || self.peek() == Some(&Token::Dot) || self.peek() == Some(&Token::LParen) {
            if self.peek() == Some(&Token::Dot) {
                self.advance()?; let prop_span = self.current_span.clone(); let prop_name = if let Some(Token::Ident(n)) = self.advance()? { n } else { return Err(self.error("Expected property", prop_span)); };
                expr = Expr::IndexAccess { target: Box::new(expr), index: Box::new(Expr::StringLiteral(prop_name)) };
            } else if self.peek() == Some(&Token::LParen) {
                let call_span = self.current_span.clone(); self.advance()?; let mut args = Vec::new(); if self.peek() != Some(&Token::RParen) { loop { args.push(self.parse_expr()?); if self.peek() == Some(&Token::Comma) { self.advance()?; } else { break; } } } self.expect(Token::RParen)?;
                if let Expr::IndexAccess { target, index } = &expr { if let (Expr::Identifier(obj), Expr::StringLiteral(prop)) = (&**target, &**index) { expr = Expr::FunctionCall { name: format!("{}.{}", obj, prop), args }; continue; } } else if let Expr::Identifier(func_name) = &expr { expr = Expr::FunctionCall { name: func_name.clone(), args }; continue; }
                return Err(self.error("Dynamic function calls are not supported yet.", call_span));
            } else {
                self.advance()?; let mut start = None; if self.peek() != Some(&Token::Colon) { start = Some(self.parse_expr()?); }

                if self.peek() == Some(&Token::Colon) {
                    self.advance()?; let mut end = None; if self.peek() != Some(&Token::RBracket) { end = Some(self.parse_expr()?); }
                    self.expect(Token::RBracket)?;
                    expr = Expr::SliceAccess { target: Box::new(expr), start: start.map(Box::new), end: end.map(Box::new) };
                } else if self.peek() == Some(&Token::Comma) {
                    let mut indices = vec![start.unwrap()];
                    while self.peek() == Some(&Token::Comma) {
                        self.advance()?;
                        indices.push(self.parse_expr()?);
                    }
                    self.expect(Token::RBracket)?;
                    expr = Expr::MultiIndexAccess { target: Box::new(expr), indices };
                } else {
                    self.expect(Token::RBracket)?;
                    expr = Expr::IndexAccess { target: Box::new(expr), index: Box::new(start.unwrap()) };
                }
            }
        } Ok(expr)
    }
}