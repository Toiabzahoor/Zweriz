// Zweriz/src/compiler.rs

use std::collections::HashMap;
use std::path::Path;
use crate::ast::{Expr, Statement};

#[derive(Debug, Clone)]
pub enum Opcode {
    LoadConst(usize, f64), LoadString(usize, String), Move(usize, usize), 
    Add(usize, usize, usize), Sub(usize, usize, usize), Mul(usize, usize, usize), 
    Div(usize, usize, usize), Mod(usize, usize, usize), Pow(usize, usize, usize), 
    BitXor(usize, usize, usize), BitAnd(usize, usize, usize), BitOr(usize, usize, usize), 
    Shl(usize, usize, usize), Shr(usize, usize, usize), BitNot(usize, usize), 
    Eq(usize, usize, usize), Lt(usize, usize, usize), Gt(usize, usize, usize), 
    And(usize, usize, usize), Or(usize, usize, usize), Not(usize, usize), Neg(usize, usize), 
    MatrixMul(usize, usize, usize), JmpIfFalse(usize, usize), Jmp(usize), Halt, 
    AllocArray(usize, usize, usize), AllocDict(usize, usize, usize), 
    LoadElement(usize, usize, usize), StoreElement(usize, usize, usize), 
    Call(usize, usize, usize, usize), NativeCall(usize, usize, usize, usize), Return(usize), 
    RunGC, Zeros(usize, usize, usize), Ones(usize, usize, usize), Len(usize, usize), Time(usize), 
    Slice(usize, usize, usize, usize), Blend(usize, usize, usize, usize), 
    Sin(usize, usize), Cos(usize, usize), Exp(usize, usize), Log(usize, usize), 
    Sqrt(usize, usize), Abs(usize, usize), Print(usize), 
    PushCatch(usize, usize), PopCatch, Throw(usize), Reduce(usize, usize, usize, u8),
    DispatchGpu { cuda_src: String, bwd_cuda_src: String, inputs: Vec<usize>, outputs: Vec<usize>, skip_pc: usize },
    LoadElementND(usize, usize, usize, usize), 
    StoreElementND(usize, usize, usize, usize),
    SetGpuMode(bool),
}

struct Scope { environment: HashMap<String, usize>, next_var_idx: usize, is_function: bool }
struct LoopContext { start_pc: usize, break_jmp_indices: Vec<usize> }

pub struct Compiler {
    scopes: Vec<Scope>, 
    pub bytecode: Vec<Opcode>, 
    functions: HashMap<String, usize>, 
    unresolved_calls: Vec<(usize, String)>, 
    loop_contexts: Vec<LoopContext>, 
    user_functions: HashMap<String, (Vec<String>, Vec<Statement>)>, 
    native_functions: HashMap<String, usize>,
}

impl Compiler {
    pub fn new() -> Self {
        let mut native_functions = HashMap::new();
        
        native_functions.insert("random.float".to_string(), 0); 
        native_functions.insert("random.uniform".to_string(), 1); 
        native_functions.insert("random.randint".to_string(), 2); 
        native_functions.insert("io.read".to_string(), 3); 
        native_functions.insert("io.write".to_string(), 4); 
        native_functions.insert("io.append".to_string(), 5); 
        native_functions.insert("io.exists".to_string(), 6); 
        native_functions.insert("mmap.load_f64".to_string(), 7); 
        native_functions.insert("os.exit".to_string(), 8); 
        native_functions.insert("os.env".to_string(), 9); 
        native_functions.insert("os.cmd".to_string(), 10); 
        native_functions.insert("time.sleep".to_string(), 11); 
        native_functions.insert("time.now".to_string(), 12); 
        native_functions.insert("math.floor".to_string(), 13); 
        native_functions.insert("math.ceil".to_string(), 14); 
        native_functions.insert("math.round".to_string(), 15); 
        native_functions.insert("math.pow".to_string(), 16); 
        native_functions.insert("string.len".to_string(), 17); 
        native_functions.insert("string.parse_num".to_string(), 18); 
        native_functions.insert("random.normal".to_string(), 19);
        
        native_functions.insert("math.transpose".to_string(), 20); 
        native_functions.insert("math.trapz".to_string(), 21); 
        native_functions.insert("math.gradient".to_string(), 22);
        native_functions.insert("math.shift2d".to_string(), 23);
        
        native_functions.insert("net.http_get".to_string(), 30); 
        native_functions.insert("net.http_post".to_string(), 31);
        native_functions.insert("net.tcp_send".to_string(), 32); 
        native_functions.insert("net.tcp_listen".to_string(), 33);
        
        // CHESS ENGINE ORCHESTRATION NATIVES
        native_functions.insert("chess_batch_generate".to_string(), 100);
        native_functions.insert("search".to_string(), 101);
        
        native_functions.insert("nn.relu".to_string(), 40); 
        native_functions.insert("nn.sigmoid".to_string(), 41);
        native_functions.insert("nn.softmax".to_string(), 42); 
        native_functions.insert("nn.track".to_string(), 43); 
        native_functions.insert("nn.backward".to_string(), 44);
        native_functions.insert("nn.step".to_string(), 45);
        native_functions.insert("nn.zero_grad".to_string(), 46);

        native_functions.insert("nn.tanh".to_string(), 47);
        native_functions.insert("nn.leaky_relu".to_string(), 48);
        native_functions.insert("nn.gelu".to_string(), 49);
        native_functions.insert("nn.dropout".to_string(), 50);
        native_functions.insert("nn.swish".to_string(), 51);
        native_functions.insert("nn.silu".to_string(), 51); 
        native_functions.insert("nn.softplus".to_string(), 52);

        native_functions.insert("io.delete".to_string(), 60);
        native_functions.insert("string.to_lower".to_string(), 61);
        native_functions.insert("string.to_upper".to_string(), 62);
        native_functions.insert("string.contains".to_string(), 63);
        native_functions.insert("array.shape".to_string(), 64);
        
        // EXTENDED UTILS
        native_functions.insert("array.push".to_string(), 65);
        native_functions.insert("array.pop".to_string(), 66);
        native_functions.insert("array.clone".to_string(), 67);
        native_functions.insert("array.concat".to_string(), 68);
        native_functions.insert("math.popcount".to_string(), 69);
        native_functions.insert("math.tzcnt".to_string(), 70);
        native_functions.insert("math.lzcnt".to_string(), 71);
        native_functions.insert("math.min".to_string(), 72);
        native_functions.insert("math.max".to_string(), 73);
        native_functions.insert("string.char_code_at".to_string(), 74);
        native_functions.insert("string.from_char_code".to_string(), 75);

        let mut comp = Self { 
            scopes: Vec::new(), 
            bytecode: Vec::new(), 
            functions: HashMap::new(), 
            unresolved_calls: Vec::new(), 
            loop_contexts: Vec::new(), 
            user_functions: HashMap::new(), 
            native_functions 
        }; 
        comp.enter_func_scope(); 
        comp
    }

    pub fn enter_block_scope(&mut self) { let next_idx = self.scopes.last().map(|s| s.next_var_idx).unwrap_or(0); self.scopes.push(Scope { environment: HashMap::new(), next_var_idx: next_idx, is_function: false }); }
    pub fn enter_func_scope(&mut self) { self.scopes.push(Scope { environment: HashMap::new(), next_var_idx: 0, is_function: true }); }
    pub fn exit_scope(&mut self) { self.scopes.pop(); }

    fn get_var_idx(&self, name: &str) -> Option<usize> { for scope in self.scopes.iter().rev() { if let Some(&idx) = scope.environment.get(name) { return Some(idx); } if scope.is_function { break; } } None }
    fn resolve_var(&mut self, name: &str) -> Result<usize, String> { if let Some(idx) = self.get_var_idx(name) { return Ok(idx); } let current = self.scopes.last_mut().unwrap(); let idx = current.next_var_idx; if idx >= 256 { return Err("Compiler Error: Register Limit Exceeded.".to_string()); } current.environment.insert(name.to_string(), idx); current.next_var_idx += 1; Ok(idx) }
    pub fn has_var(&self, name: &str) -> bool { self.get_var_idx(name).is_some() }

    pub fn compile(&mut self, ast: Vec<Statement>) -> Result<Vec<Opcode>, String> {
        for stmt in ast { self.compile_statement(stmt)?; } self.bytecode.push(Opcode::Halt);
        for (idx, name) in &self.unresolved_calls { let target = *self.functions.get(name).ok_or(format!("Undefined function '{}'", name))?; if let Opcode::Call(_, dest, start, count) = self.bytecode[*idx] { self.bytecode[*idx] = Opcode::Call(target, dest, start, count); } }
        Ok(self.bytecode.clone())
    }

    pub fn compile_line(&mut self, ast: Vec<Statement>) -> Result<(), String> {
        for stmt in ast { self.compile_statement(stmt)?; } self.bytecode.push(Opcode::Halt); Ok(())
    }

    fn is_gpu_compatible_expr(&self, expr: &Expr) -> bool {
        match expr { 
            Expr::Number(_) | Expr::Boolean(_) | Expr::Identifier(_) => true, 
            Expr::BinaryOp { left, right, .. } => self.is_gpu_compatible_expr(left) && self.is_gpu_compatible_expr(right), 
            Expr::MatrixOp { .. } => false,
            Expr::UnaryOp { right, .. } => self.is_gpu_compatible_expr(right), 
            Expr::FunctionCall { name, args } => { 
                let math_funcs = ["sin", "cos", "exp", "log", "sqrt", "abs", "blend", "zeros", "ones", "math.floor", "math.ceil", "math.round"]; 
                if math_funcs.contains(&name.as_str()) { return args.iter().all(|a| self.is_gpu_compatible_expr(a)); } 
                if ["sum", "mean", "max", "min"].contains(&name.as_str()) { return false; } 
                if let Some((_, body)) = self.user_functions.get(name) { return args.iter().all(|a| self.is_gpu_compatible_expr(a)) && body.iter().all(|s| self.is_gpu_compatible_stmt(s)); } 
                false 
            } 
            _ => false, 
        }
    }

    fn is_gpu_compatible_stmt(&self, stmt: &Statement) -> bool {
        match stmt { 
            Statement::Assignment { value, .. } => self.is_gpu_compatible_expr(value), 
            Statement::If { condition, then_branch, else_branch } => { if !self.is_gpu_compatible_expr(condition) { return false; } then_branch.iter().all(|s| self.is_gpu_compatible_stmt(s)) && else_branch.as_ref().map_or(true, |b| b.iter().all(|s| self.is_gpu_compatible_stmt(s))) } 
            Statement::While { condition, body } => self.is_gpu_compatible_expr(condition) && body.iter().all(|s| self.is_gpu_compatible_stmt(s)), 
            Statement::For { start, end, body, .. } => self.is_gpu_compatible_expr(start) && self.is_gpu_compatible_expr(end) && body.iter().all(|s| self.is_gpu_compatible_stmt(s)), 
            Statement::ForEach { .. } => false, 
            Statement::Return { value } => self.is_gpu_compatible_expr(value), Statement::Break | Statement::Continue => true, 
            _ => false, 
        }
    }

    fn get_written_vars(stmt: &Statement, writes: &mut Vec<String>) {
        match stmt { Statement::Assignment { name, .. } => writes.push(name.clone()), Statement::IndexAssignment { target, .. } => { if let Expr::Identifier(n) = target { writes.push(n.clone()); } } Statement::MultiIndexAssignment { target, .. } => { if let Expr::Identifier(n) = target { writes.push(n.clone()); } } Statement::If { then_branch, else_branch, .. } => { for s in then_branch { Self::get_written_vars(s, writes); } if let Some(eb) = else_branch { for s in eb { Self::get_written_vars(s, writes); } } } Statement::While { body, .. } => { for s in body { Self::get_written_vars(s, writes); } } Statement::For { iterator, body, .. } | Statement::ForEach { iterator, body, .. } => { writes.push(iterator.clone()); for s in body { Self::get_written_vars(s, writes); } } _ => {} }
    }

    fn get_read_vars_expr(expr: &Expr, reads: &mut Vec<String>) {
        match expr { Expr::Identifier(name) => reads.push(name.clone()), Expr::BinaryOp { left, right, .. } | Expr::MatrixOp { left, right, .. } => { Self::get_read_vars_expr(left, reads); Self::get_read_vars_expr(right, reads); } Expr::UnaryOp { right, .. } => Self::get_read_vars_expr(right, reads), Expr::FunctionCall { args, .. } => { for arg in args { Self::get_read_vars_expr(arg, reads); } } Expr::Array(elements) => { for el in elements { Self::get_read_vars_expr(el, reads); } } Expr::Dictionary(pairs) => { for (k, v) in pairs { Self::get_read_vars_expr(k, reads); Self::get_read_vars_expr(v, reads); } } Expr::SliceAccess { target, start, end } => { Self::get_read_vars_expr(target, reads); if let Some(s) = start { Self::get_read_vars_expr(s, reads); } if let Some(e) = end { Self::get_read_vars_expr(e, reads); } } Expr::IndexAccess { target, index } => { Self::get_read_vars_expr(target, reads); Self::get_read_vars_expr(index, reads); } Expr::MultiIndexAccess { target, indices } => { Self::get_read_vars_expr(target, reads); for i in indices { Self::get_read_vars_expr(i, reads); } } _ => {} }
    }

    fn get_read_vars(stmt: &Statement, reads: &mut Vec<String>) {
        match stmt { Statement::Expression(expr) | Statement::Return { value: expr } => Self::get_read_vars_expr(expr, reads), Statement::Assignment { value, .. } => Self::get_read_vars_expr(value, reads), Statement::IndexAssignment { index, value, .. } => { Self::get_read_vars_expr(index, reads); Self::get_read_vars_expr(value, reads); } Statement::MultiIndexAssignment { target, indices, value } => { Self::get_read_vars_expr(target, reads); for i in indices { Self::get_read_vars_expr(i, reads); } Self::get_read_vars_expr(value, reads); } Statement::Print { value } => Self::get_read_vars_expr(value, reads), Statement::If { condition, then_branch, else_branch } => { Self::get_read_vars_expr(condition, reads); for s in then_branch { Self::get_read_vars(s, reads); } if let Some(eb) = else_branch { for s in eb { Self::get_read_vars(s, reads); } } } Statement::While { condition, body } => { Self::get_read_vars_expr(condition, reads); for s in body { Self::get_read_vars(s, reads); } } Statement::For { start, end, body, .. } => { Self::get_read_vars_expr(start, reads); Self::get_read_vars_expr(end, reads); for s in body { Self::get_read_vars(s, reads); } } Statement::ForEach { array, body, .. } => { Self::get_read_vars_expr(array, reads); for s in body { Self::get_read_vars(s, reads); } } _ => {} }
    }

    fn transpile_expr(&self, expr: &Expr, inputs: &mut Vec<String>, outputs: &mut Vec<String>, scalars: &mut Vec<String>, req_funcs: &mut Vec<String>) -> Result<String, String> {
        match expr { 
            Expr::Number(n) => Ok(if n.fract() == 0.0 { format!("((REAL_T){}.0)", n) } else { format!("((REAL_T){})", n) }), 
            Expr::Boolean(b) => Ok(if *b { "((REAL_T)1.0)".to_string() } else { "((REAL_T)0.0)".to_string() }), 
            Expr::Identifier(name) => { 
                if outputs.contains(name) || (self.has_var(name) && !scalars.contains(name)) { 
                    if !inputs.contains(name) && !outputs.contains(name) { inputs.push(name.clone()); } 
                    Ok(format!("((REAL_T){}[index])", name)) 
                } else { 
                    if !scalars.contains(name) { scalars.push(name.clone()); } 
                    Ok(name.clone()) 
                } 
            } 
            Expr::UnaryOp { op, right } => { let r = self.transpile_expr(right, inputs, outputs, scalars, req_funcs)?; if op == "not" { Ok(format!("(!{})", r)) } else if op == "-" { Ok(format!("(-{})", r)) } else if op == "~" { Ok(format!("(REAL_T)(~(int64_t){})", r)) } else { Err(format!("Unsupported unary op: {}", op)) } } 
            Expr::BinaryOp { left, op, right } => { let l = self.transpile_expr(left, inputs, outputs, scalars, req_funcs)?; let r = self.transpile_expr(right, inputs, outputs, scalars, req_funcs)?; Ok(match op.as_str() { "and" => format!("({} && {})", l, r), "or" => format!("({} || {})", l, r), "%" => format!("fmod({}, {})", l, r), "**" => format!("pow({}, {})", l, r), "^" => format!("(REAL_T)((int64_t){} ^ (int64_t){})", l, r), "|" => format!("(REAL_T)((int64_t){} | (int64_t){})", l, r), "&" => format!("(REAL_T)((int64_t){} & (int64_t){})", l, r), "<<" => format!("(REAL_T)((int64_t){} << (int64_t){})", l, r), ">>" => format!("(REAL_T)((int64_t){} >> (int64_t){})", l, r), _ => format!("({} {} {})", l, op.as_str(), r) }) } 
            Expr::FunctionCall { name, args } => { 
                if name == "zeros" { return Ok("((REAL_T)0.0)".to_string()); } 
                if name == "ones" { return Ok("((REAL_T)1.0)".to_string()); } 
                let mut arg_strs = Vec::new(); 
                for arg in args { arg_strs.push(self.transpile_expr(arg, inputs, outputs, scalars, req_funcs)?); } 
                
                let math_funcs = ["sin", "cos", "exp", "log", "sqrt", "abs", "math.floor", "math.ceil", "math.round"]; 
                if math_funcs.contains(&name.as_str()) { 
                    let c_name = name.replace("math.", "");
                    return Ok(format!("{}({})", c_name, arg_strs.join(", "))); 
                } 
                
                if name == "blend" { return Ok(format!("(({}) != 0.0 ? ({}) : ({}))", arg_strs[0], arg_strs[1], arg_strs[2])); } 
                if self.user_functions.contains_key(name) { 
                    if !req_funcs.contains(name) { req_funcs.push(name.clone()); } 
                    return Ok(format!("zweriz_usr_{}({})", name, arg_strs.join(", "))); 
                } 
                Err(format!("Transpiler Error: Function '{}' not supported on GPU.", name)) 
            } 
            _ => Err("Transpiler Error: Unsupported expression inside GPU Block.".to_string()), 
        }
    }

    fn transpile_stmt(&self, stmt: &Statement, inputs: &mut Vec<String>, outputs: &mut Vec<String>, scalars: &mut Vec<String>, req_funcs: &mut Vec<String>) -> Result<String, String> {
        match stmt { 
            Statement::Assignment { name, value } => { let val_str = self.transpile_expr(value, inputs, outputs, scalars, req_funcs)?; let is_constant_init = matches!(value, Expr::Number(_) | Expr::Boolean(_)); if !scalars.contains(name) && (outputs.contains(name) || self.has_var(name) || !is_constant_init) { if !outputs.contains(name) { outputs.push(name.clone()); } Ok(format!("    {}[index] = (float)({});\n", name, val_str)) } else { if !scalars.contains(name) { scalars.push(name.clone()); Ok(format!("    REAL_T {} = {};\n", name, val_str)) } else { Ok(format!("    {} = {};\n", name, val_str)) } } } 
            Statement::While { condition, body } => { let cond_str = self.transpile_expr(condition, inputs, outputs, scalars, req_funcs)?; let mut body_str = String::new(); for s in body { body_str.push_str(&self.transpile_stmt(s, inputs, outputs, scalars, req_funcs)?); } Ok(format!("    while ({}) {{\n{}}}\n", cond_str, body_str)) } 
            Statement::For { iterator, start, end, body, .. } => { let start_str = self.transpile_expr(start, inputs, outputs, scalars, req_funcs)?; let end_str = self.transpile_expr(end, inputs, outputs, scalars, req_funcs)?; if !scalars.contains(iterator) { scalars.push(iterator.clone()); } let mut body_str = String::new(); for s in body { body_str.push_str(&self.transpile_stmt(s, inputs, outputs, scalars, req_funcs)?); } Ok(format!("    for (REAL_T {} = {}; {} < {}; {}++) {{\n{}}}\n", iterator, start_str, iterator, end_str, iterator, body_str)) }
            Statement::If { condition, then_branch, else_branch } => { let cond_str = self.transpile_expr(condition, inputs, outputs, scalars, req_funcs)?; let mut then_str = String::new(); for s in then_branch { then_str.push_str(&self.transpile_stmt(s, inputs, outputs, scalars, req_funcs)?); } let mut cpp_block = format!("    if ({}) {{\n{}    }}\n", cond_str, then_str); if let Some(else_stmts) = else_branch { let mut else_str = String::new(); for s in else_stmts { else_str.push_str(&self.transpile_stmt(s, inputs, outputs, scalars, req_funcs)?); } cpp_block.push_str(&format!("    else {{\n{}    }}\n", else_str)); } Ok(cpp_block) } 
            Statement::Return { value } => Ok(format!("    return {};\n", self.transpile_expr(value, inputs, outputs, scalars, req_funcs)?)), Statement::Break => Ok("    break;\n".to_string()), Statement::Continue => Ok("    continue;\n".to_string()), 
            _ => Err("Transpiler Error: Unsupported statement inside GPU Block.".to_string()), 
        }
    }

    fn compile_single_gpu_kernel(&mut self, statements: &[Statement]) -> Result<(), String> {
        let mut inputs = Vec::new(); let mut outputs = Vec::new(); let mut scalars = Vec::new(); let mut req_funcs = Vec::new(); let mut cpp_body = String::new();
        for s in statements { cpp_body.push_str(&self.transpile_stmt(s, &mut inputs, &mut outputs, &mut scalars, &mut req_funcs)?); } 
        inputs.retain(|x| !outputs.contains(x));

        let mut cuda_src = String::new(); cuda_src.push_str("typedef float REAL_T;\n");
        let mut processed_funcs: Vec<String> = Vec::new();
        while let Some(f_name) = req_funcs.iter().find(|f| !processed_funcs.contains(*f)).cloned() { processed_funcs.push(f_name.clone()); let (params, body) = self.user_functions.get(&f_name).unwrap().clone(); let mut f_inputs = Vec::new(); let mut f_outputs = Vec::new(); let mut f_scalars = params.clone(); let mut f_body = String::new(); for s in &body { f_body.push_str(&self.transpile_stmt(s, &mut f_inputs, &mut f_outputs, &mut f_scalars, &mut req_funcs)?); } let param_str = params.iter().map(|p| format!("REAL_T {}", p)).collect::<Vec<_>>().join(", "); cuda_src.push_str(&format!("extern \"C\" __device__ REAL_T zweriz_usr_{}({}) {{\n{}}}\n\n", f_name, param_str, f_body)); }
        
        cuda_src.push_str("extern \"C\" __global__ void zweriz_kernel(const unsigned long long* __in_ptrs, unsigned long long* __out_ptrs, const unsigned long long* __shapes, size_t size) {\n    size_t index = blockIdx.x * blockDim.x + threadIdx.x;\n    if (index >= size) return;\n");
        for (i, inp) in inputs.iter().enumerate() { cuda_src.push_str(&format!("    const float* {} = (const float*)__in_ptrs[{}];\n", inp, i)); } for (i, out) in outputs.iter().enumerate() { cuda_src.push_str(&format!("    float* {} = (float*)__out_ptrs[{}];\n", out, i)); } cuda_src.push_str(&cpp_body); cuda_src.push_str("}\n");
        
        let mut input_indices = Vec::new(); for inp in &inputs { if !self.has_var(inp) { self.resolve_var(inp)?; } input_indices.push(self.resolve_var(inp)?); } 
        let mut output_indices = Vec::new(); for out in &outputs { if !self.has_var(out) { self.resolve_var(out)?; } output_indices.push(self.resolve_var(out)?); }
        
        let dispatch_idx = self.bytecode.len(); 
        self.bytecode.push(Opcode::DispatchGpu { cuda_src, bwd_cuda_src: String::new(), inputs: input_indices, outputs: output_indices, skip_pc: 0 }); 

        for s in statements { self.compile_statement(s.clone())?; }
        let end_idx = self.bytecode.len(); 
        if let Opcode::DispatchGpu { ref mut skip_pc, .. } = self.bytecode[dispatch_idx] { *skip_pc = end_idx; }
        Ok(())
    }

    fn compile_gpu_block(&mut self, statements: &[Statement]) -> Result<(), String> {
        let mut pre_cpu_stmts = Vec::new(); let mut gpu_stmts = Vec::new(); let mut post_cpu_stmts = Vec::new(); let mut gpu_indices = Vec::new();
        for (i, stmt) in statements.iter().enumerate() { if self.is_gpu_compatible_stmt(stmt) { gpu_stmts.push(stmt.clone()); gpu_indices.push(i); } }
        let last_gpu_index = if gpu_indices.is_empty() { statements.len() } else { *gpu_indices.last().unwrap() }; let mut gpu_writes = Vec::new(); for stmt in &gpu_stmts { Self::get_written_vars(stmt, &mut gpu_writes); } let mut has_pushed_to_post = false;
        for (i, stmt) in statements.iter().enumerate() { if self.is_gpu_compatible_stmt(stmt) { continue; } let mut reads = Vec::new(); Self::get_read_vars(stmt, &mut reads); let depends_on_gpu = reads.iter().any(|r| gpu_writes.contains(r)); if depends_on_gpu || i > last_gpu_index || has_pushed_to_post { post_cpu_stmts.push(stmt.clone()); has_pushed_to_post = true; } else { pre_cpu_stmts.push(stmt.clone()); } }
        for stmt in pre_cpu_stmts { self.compile_statement(stmt)?; } if !gpu_stmts.is_empty() { self.compile_single_gpu_kernel(&gpu_stmts)?; } for stmt in post_cpu_stmts { self.compile_statement(stmt)?; }
        Ok(())
    }

    pub fn compile_statement(&mut self, stmt: Statement) -> Result<(), String> {
        let current_temp = self.scopes.last().unwrap().next_var_idx;
        match stmt {
            Statement::Expression(expr) => { self.compile_expression(expr, current_temp, current_temp + 1)?; }
            Statement::Break => { let jmp_idx = self.bytecode.len(); self.bytecode.push(Opcode::Jmp(0)); self.loop_contexts.last_mut().unwrap().break_jmp_indices.push(jmp_idx); }
            Statement::Continue => { let start_pc = self.loop_contexts.last().unwrap().start_pc; self.bytecode.push(Opcode::Jmp(start_pc)); }
            Statement::Import { module } => { 
                let filename = format!("{}.zw", module); let path = Path::new(&filename);
                if !path.exists() { return Err(format!("Compiler Error: Cannot import module '{}'. File '{}' not found.", module, filename)); }
                let source = std::fs::read_to_string(path).map_err(|_| format!("Compiler Error: Failed to read file '{}'", filename))?; 
                let mut parser = crate::parser::Parser::new(&source); let ast = parser.parse()?; for s in ast { self.compile_statement(s)?; } 
            }
            Statement::For { iterator, start, end, body } => {
                self.enter_block_scope(); let iter_reg = self.resolve_var(&iterator)?; let end_reg = self.resolve_var("..end")?; let cond_reg = self.resolve_var("..cond")?; let one_reg = self.resolve_var("..one")?;
                self.compile_expression(start, iter_reg, self.scopes.last().unwrap().next_var_idx)?; self.compile_expression(end, end_reg, self.scopes.last().unwrap().next_var_idx)?;
                let loop_start = self.bytecode.len(); self.loop_contexts.push(LoopContext { start_pc: loop_start, break_jmp_indices: Vec::new() });
                self.bytecode.push(Opcode::Lt(cond_reg, iter_reg, end_reg)); let jmp_false_idx = self.bytecode.len(); self.bytecode.push(Opcode::JmpIfFalse(cond_reg, 0));
                for s in body { self.compile_statement(s)?; }
                self.bytecode.push(Opcode::LoadConst(one_reg, 1.0)); self.bytecode.push(Opcode::Add(iter_reg, iter_reg, one_reg)); self.bytecode.push(Opcode::Jmp(loop_start));
                let end_pc = self.bytecode.len(); self.bytecode[jmp_false_idx] = Opcode::JmpIfFalse(cond_reg, end_pc); let ctx = self.loop_contexts.pop().unwrap(); for break_idx in ctx.break_jmp_indices { self.bytecode[break_idx] = Opcode::Jmp(end_pc); } self.exit_scope();
            }
            Statement::ForEach { iterator, array, body } => {
                self.enter_block_scope(); let arr_reg = self.resolve_var("..arr")?; let len_reg = self.resolve_var("..len")?; let idx_reg = self.resolve_var("..idx")?; let cond_reg = self.resolve_var("..cond")?; let one_reg = self.resolve_var("..one")?; let iter_reg = self.resolve_var(&iterator)?;
                self.compile_expression(array, arr_reg, self.scopes.last().unwrap().next_var_idx)?; self.bytecode.push(Opcode::Len(len_reg, arr_reg)); self.bytecode.push(Opcode::LoadConst(idx_reg, 0.0));
                let loop_start = self.bytecode.len(); self.loop_contexts.push(LoopContext { start_pc: loop_start, break_jmp_indices: Vec::new() });
                self.bytecode.push(Opcode::Lt(cond_reg, idx_reg, len_reg)); let jmp_false_idx = self.bytecode.len(); self.bytecode.push(Opcode::JmpIfFalse(cond_reg, 0));
                self.bytecode.push(Opcode::LoadElement(iter_reg, arr_reg, idx_reg)); for s in body { self.compile_statement(s)?; }
                self.bytecode.push(Opcode::LoadConst(one_reg, 1.0)); self.bytecode.push(Opcode::Add(idx_reg, idx_reg, one_reg)); self.bytecode.push(Opcode::Jmp(loop_start));
                let end_pc = self.bytecode.len(); self.bytecode[jmp_false_idx] = Opcode::JmpIfFalse(cond_reg, end_pc); let ctx = self.loop_contexts.pop().unwrap(); for break_idx in ctx.break_jmp_indices { self.bytecode[break_idx] = Opcode::Jmp(end_pc); } self.exit_scope();
            }
            Statement::FunctionDecl { name, params, body } => { self.user_functions.insert(name.clone(), (params.clone(), body.clone())); let jmp_over_idx = self.bytecode.len(); self.bytecode.push(Opcode::Jmp(0)); let func_pc = self.bytecode.len(); self.functions.insert(name.clone(), func_pc); self.enter_func_scope(); for param in params { self.resolve_var(&param)?; } for s in body { self.compile_statement(s)?; } self.bytecode.push(Opcode::LoadConst(255, 0.0)); self.bytecode.push(Opcode::Return(255)); self.exit_scope(); self.bytecode[jmp_over_idx] = Opcode::Jmp(self.bytecode.len()); }
            Statement::Return { value } => { let ret_reg = current_temp; self.compile_expression(value, ret_reg, current_temp + 1)?; self.bytecode.push(Opcode::Return(ret_reg)); }
            Statement::Assignment { name, value } => { let var_idx = self.resolve_var(&name)?; self.compile_expression(value, var_idx, current_temp)?; }
            Statement::IndexAssignment { target, index, value } => { let target_reg = current_temp; let index_reg = current_temp + 1; let val_reg = current_temp + 2; self.compile_expression(target, target_reg, current_temp + 3)?; self.compile_expression(index, index_reg, current_temp + 3)?; self.compile_expression(value, val_reg, current_temp + 3)?; self.bytecode.push(Opcode::StoreElement(target_reg, index_reg, val_reg)); }
            Statement::MultiIndexAssignment { target, indices, value } => { 
                let target_reg = current_temp; self.compile_expression(target, target_reg, current_temp + 2)?; 
                let val_reg = current_temp + 1; self.compile_expression(value, val_reg, current_temp + 2)?; 
                let count = indices.len(); let indices_start = current_temp + 2; 
                for (i, idx_expr) in indices.into_iter().enumerate() { self.compile_expression(idx_expr, indices_start + i, indices_start + count)?; } 
                self.bytecode.push(Opcode::StoreElementND(target_reg, indices_start, count, val_reg)); 
            }
            Statement::Print { value } => { let val_reg = current_temp; self.compile_expression(value, val_reg, current_temp + 1)?; self.bytecode.push(Opcode::Print(val_reg)); }
            Statement::If { condition, then_branch, else_branch } => { let cond_reg = current_temp; self.compile_expression(condition, cond_reg, current_temp + 1)?; let jmp_false_idx = self.bytecode.len(); self.bytecode.push(Opcode::JmpIfFalse(cond_reg, 0)); for s in then_branch { self.compile_statement(s)?; } if let Some(else_stmts) = else_branch { let jmp_end_idx = self.bytecode.len(); self.bytecode.push(Opcode::Jmp(0)); self.bytecode[jmp_false_idx] = Opcode::JmpIfFalse(cond_reg, self.bytecode.len()); for s in else_stmts { self.compile_statement(s)?; } self.bytecode[jmp_end_idx] = Opcode::Jmp(self.bytecode.len()); } else { self.bytecode[jmp_false_idx] = Opcode::JmpIfFalse(cond_reg, self.bytecode.len()); } }
            Statement::While { condition, body } => { let loop_start = self.bytecode.len(); self.loop_contexts.push(LoopContext { start_pc: loop_start, break_jmp_indices: Vec::new() }); let cond_reg = current_temp; self.compile_expression(condition, cond_reg, current_temp + 1)?; let jmp_false_idx = self.bytecode.len(); self.bytecode.push(Opcode::JmpIfFalse(cond_reg, 0)); for s in body { self.compile_statement(s)?; } self.bytecode.push(Opcode::Jmp(loop_start)); let end_pc = self.bytecode.len(); self.bytecode[jmp_false_idx] = Opcode::JmpIfFalse(cond_reg, end_pc); let ctx = self.loop_contexts.pop().unwrap(); for break_idx in ctx.break_jmp_indices { self.bytecode[break_idx] = Opcode::Jmp(end_pc); } }
            Statement::GpuBlock { statements } => { 
                self.bytecode.push(Opcode::SetGpuMode(true));
                self.compile_gpu_block(&statements)?; 
                self.bytecode.push(Opcode::SetGpuMode(false));
            } 
            Statement::TryCatch { try_block, error_var, catch_block } => { let push_idx = self.bytecode.len(); self.bytecode.push(Opcode::PushCatch(0, 0)); for s in try_block { self.compile_statement(s)?; } self.bytecode.push(Opcode::PopCatch); let jmp_over_catch = self.bytecode.len(); self.bytecode.push(Opcode::Jmp(0)); let catch_pc = self.bytecode.len(); self.enter_block_scope(); let err_reg = self.resolve_var(&error_var)?; self.bytecode[push_idx] = Opcode::PushCatch(catch_pc, err_reg); for s in catch_block { self.compile_statement(s)?; } self.exit_scope(); self.bytecode[jmp_over_catch] = Opcode::Jmp(self.bytecode.len()); }
            Statement::Throw { value } => { self.compile_expression(value, current_temp, current_temp + 1)?; self.bytecode.push(Opcode::Throw(current_temp)); }
        } Ok(())
    }

    fn compile_expression(&mut self, expr: Expr, dest: usize, current_temp: usize) -> Result<(), String> {
        match expr {
            Expr::Number(val) => self.bytecode.push(Opcode::LoadConst(dest, val)), Expr::Boolean(val) => self.bytecode.push(Opcode::LoadConst(dest, if val { 1.0 } else { 0.0 })), Expr::StringLiteral(text) => self.bytecode.push(Opcode::LoadString(dest, text)), Expr::Identifier(name) => { let idx = self.resolve_var(&name)?; if idx != dest { self.bytecode.push(Opcode::Move(dest, idx)); } }
            Expr::UnaryOp { op, right } => { let r_reg = current_temp; self.compile_expression(*right, r_reg, current_temp + 1)?; if op == "not" { self.bytecode.push(Opcode::Not(dest, r_reg)); } else if op == "-" { self.bytecode.push(Opcode::Neg(dest, r_reg)); } else if op == "~" { self.bytecode.push(Opcode::BitNot(dest, r_reg)); } }
            Expr::FunctionCall { name, args } => {
                if name == "gc" { self.bytecode.push(Opcode::RunGC); return Ok(()); } if name == "time" { self.bytecode.push(Opcode::Time(dest)); return Ok(()); }
                
                if name == "sum" || name == "mean" || name == "max" || name == "min" {
                    let op_type = match name.as_str() { "sum" => 0, "mean" => 1, "max" => 2, "min" => 3, _ => 0 };
                    self.compile_expression(args[0].clone(), current_temp, current_temp + 2)?;
                    if args.len() == 2 {
                        self.compile_expression(args[1].clone(), current_temp + 1, current_temp + 2)?;
                    } else {
                        self.bytecode.push(Opcode::LoadConst(current_temp + 1, -1.0));
                    }
                    self.bytecode.push(Opcode::Reduce(dest, current_temp, current_temp + 1, op_type));
                    return Ok(());
                }

                if let Some(&native_id) = self.native_functions.get(&name) { let arg_count = args.len(); let arg_start = current_temp; for (i, arg) in args.into_iter().enumerate() { self.compile_expression(arg, arg_start + i, arg_start + arg_count)?; } self.bytecode.push(Opcode::NativeCall(native_id, dest, arg_start, arg_count)); return Ok(()); }
                if name == "zeros" || name == "ones" { let (r_reg, c_reg) = if args.len() == 1 { self.compile_expression(args[0].clone(), current_temp, current_temp + 2)?; self.bytecode.push(Opcode::LoadConst(current_temp + 1, 1.0)); (current_temp, current_temp + 1) } else if args.len() == 2 { self.compile_expression(args[0].clone(), current_temp, current_temp + 2)?; self.compile_expression(args[1].clone(), current_temp + 1, current_temp + 2)?; (current_temp, current_temp + 1) } else { return Err("zeros/ones requires 1 or 2 arguments".to_string()); }; self.bytecode.push(if name == "zeros" { Opcode::Zeros(dest, r_reg, c_reg) } else { Opcode::Ones(dest, r_reg, c_reg) }); return Ok(()); }
                if name == "len" { self.compile_expression(args[0].clone(), current_temp, current_temp + 1)?; self.bytecode.push(Opcode::Len(dest, current_temp)); return Ok(()); } if name == "blend" { self.compile_expression(args[0].clone(), current_temp, current_temp + 3)?; self.compile_expression(args[1].clone(), current_temp + 1, current_temp + 3)?; self.compile_expression(args[2].clone(), current_temp + 2, current_temp + 3)?; self.bytecode.push(Opcode::Blend(dest, current_temp, current_temp + 1, current_temp + 2)); return Ok(()); }
                let math_funcs = ["sin", "cos", "exp", "log", "sqrt", "abs"]; if math_funcs.contains(&name.as_str()) { self.compile_expression(args[0].clone(), current_temp, current_temp + 1)?; let op = match name.as_str() { "sin" => Opcode::Sin(dest, current_temp), "cos" => Opcode::Cos(dest, current_temp), "exp" => Opcode::Exp(dest, current_temp), "log" => Opcode::Log(dest, current_temp), "sqrt" => Opcode::Sqrt(dest, current_temp), _ => Opcode::Abs(dest, current_temp) }; self.bytecode.push(op); return Ok(()); }
                let arg_count = args.len(); let arg_start = current_temp; for (i, arg) in args.into_iter().enumerate() { self.compile_expression(arg, arg_start + i, arg_start + arg_count)?; } let call_idx = self.bytecode.len(); self.bytecode.push(Opcode::Call(0, dest, arg_start, arg_count)); self.unresolved_calls.push((call_idx, name));
            }
            Expr::Array(elements) => { let size = elements.len(); let elem_start = current_temp; for (i, el) in elements.into_iter().enumerate() { self.compile_expression(el, elem_start + i, elem_start + size)?; } self.bytecode.push(Opcode::AllocArray(dest, size, elem_start)); }
            Expr::Dictionary(pairs) => { let size = pairs.len(); let start_reg = current_temp; for (i, (k, v)) in pairs.into_iter().enumerate() { self.compile_expression(k, start_reg + i * 2, start_reg + size * 2)?; self.compile_expression(v, start_reg + i * 2 + 1, start_reg + size * 2)?; } self.bytecode.push(Opcode::AllocDict(dest, size, start_reg)); }
            Expr::SliceAccess { target, start, end } => { self.compile_expression(*target, current_temp, current_temp + 3)?; if let Some(s) = start { self.compile_expression(*s, current_temp + 1, current_temp + 3)?; } else { self.bytecode.push(Opcode::LoadConst(current_temp + 1, 0.0)); } if let Some(e) = end { self.compile_expression(*e, current_temp + 2, current_temp + 3)?; } else { self.bytecode.push(Opcode::LoadConst(current_temp + 2, -1.0)); } self.bytecode.push(Opcode::Slice(dest, current_temp, current_temp + 1, current_temp + 2)); }
            Expr::IndexAccess { target, index } => { self.compile_expression(*target, current_temp, current_temp + 2)?; self.compile_expression(*index, current_temp + 1, current_temp + 2)?; self.bytecode.push(Opcode::LoadElement(dest, current_temp, current_temp + 1)); }
            Expr::MultiIndexAccess { target, indices } => { 
                let target_reg = current_temp; self.compile_expression(*target, target_reg, current_temp + 1)?; 
                let count = indices.len(); let indices_start = current_temp + 1; 
                for (i, idx_expr) in indices.into_iter().enumerate() { self.compile_expression(idx_expr, indices_start + i, indices_start + count)?; } 
                self.bytecode.push(Opcode::LoadElementND(dest, target_reg, indices_start, count)); 
            }
            Expr::BinaryOp { left, op, right } => {
                self.compile_expression(*left, current_temp, current_temp + 2)?; self.compile_expression(*right, current_temp + 1, current_temp + 2)?;
                match op.as_str() { "*" => self.bytecode.push(Opcode::Mul(dest, current_temp, current_temp + 1)), "+" => self.bytecode.push(Opcode::Add(dest, current_temp, current_temp + 1)), "-" => self.bytecode.push(Opcode::Sub(dest, current_temp, current_temp + 1)), "/" => self.bytecode.push(Opcode::Div(dest, current_temp, current_temp + 1)), "%" => self.bytecode.push(Opcode::Mod(dest, current_temp, current_temp + 1)), "**" => self.bytecode.push(Opcode::Pow(dest, current_temp, current_temp + 1)), "^" => self.bytecode.push(Opcode::BitXor(dest, current_temp, current_temp + 1)), "|" => self.bytecode.push(Opcode::BitOr(dest, current_temp, current_temp + 1)), "&" => self.bytecode.push(Opcode::BitAnd(dest, current_temp, current_temp + 1)), "<<" => self.bytecode.push(Opcode::Shl(dest, current_temp, current_temp + 1)), ">>" => self.bytecode.push(Opcode::Shr(dest, current_temp, current_temp + 1)), "==" => self.bytecode.push(Opcode::Eq(dest, current_temp, current_temp + 1)), "<" => self.bytecode.push(Opcode::Lt(dest, current_temp, current_temp + 1)), ">" => self.bytecode.push(Opcode::Gt(dest, current_temp, current_temp + 1)), "and" => self.bytecode.push(Opcode::And(dest, current_temp, current_temp + 1)), "or" => self.bytecode.push(Opcode::Or(dest, current_temp, current_temp + 1)), _ => return Err(format!("Operator {}", op)), }
            }
            Expr::MatrixOp { left, op: _, right } => { self.compile_expression(*left, current_temp, current_temp + 2)?; self.compile_expression(*right, current_temp + 1, current_temp + 2)?; self.bytecode.push(Opcode::MatrixMul(dest, current_temp, current_temp + 1)); }
        } Ok(())
    }
}