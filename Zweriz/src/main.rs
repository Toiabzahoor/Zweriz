pub mod ast;
pub mod lexer;
pub mod parser;
pub mod compiler;
pub mod vm;
pub mod gpu;
pub mod modules;

use std::fs;
use std::env;
use std::io::{self, Write};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 {

        let filename = &args[1];

        println!("[Zweriz Interpreter] Running '{}' - made by toiabzahoor", filename);

        let source = match fs::read_to_string(filename) {
            Ok(s) => s,
            Err(e) => { eprintln!("🚨 Error: Failed to read file '{}': {}", filename, e); std::process::exit(1); }
        };

        let mut parser = parser::Parser::new(&source);
        match parser.parse() {
            Ok(ast) => {
                let mut compiler = compiler::Compiler::new();
                match compiler.compile(ast) {
                    Ok(program) => {
                        let mut vm = vm::Vm::new();
                        if let Err(e) = vm.execute(&program) {
                            eprintln!("\n🚨 Zweriz Runtime Error :\n   --> Uncaught Exception: {}", e);
                            std::process::exit(1);
                        }
                    }
                    Err(e) => { eprintln!("\n🚨 Compilation Error :\n   --> {}", e); std::process::exit(1); }
                }
            }
            Err(e) => { eprintln!("\n🚨 Syntax Error (made by toiabzahoor):\n{}", e); std::process::exit(1); }
        }
    } else {

        println!("======================================");
        println!(" Zweriz Interactive REPL v0.2");
        println!(" made by toiabzahoor");
        println!(" Type 'exit' to quit or 'help' for guide.");
        println!("======================================");

        let mut vm = vm::Vm::new();
        let mut compiler = compiler::Compiler::new();
        let mut input_buffer = String::new();

        loop {
            if input_buffer.is_empty() { print!(">>> "); } else { print!("... "); }
            io::stdout().flush().unwrap();

            let mut line = String::new();
            if io::stdin().read_line(&mut line).is_err() { break; }
            let line = line.trim();

            if input_buffer.is_empty() && (line == "exit" || line == "quit") { break; }
            if input_buffer.is_empty() && line == "help" {
                println!("======================================");
                println!(" Zweriz Help Guide (made by toiabzahoor)");
                println!("======================================");
                println!(" 1. Variables : x = 10, y = [1, 2, 3]");
                println!(" 2. Math      : a = sin(x) + cos(x)");
                println!(" 3. N-D Array : arr = [[1, 2], [3, 4]]");
                println!(" 4. Indexing  : val = arr[0, 1]");
                println!(" 5. GPU Block : GPU {{ arr = arr * 2 }}");
                println!(" 6. Print     : print(arr)");
                println!("======================================");
                continue;
            }

            input_buffer.push_str(line);
            input_buffer.push('\n');

            let open_braces = input_buffer.chars().filter(|&c| c == '{').count();
            let close_braces = input_buffer.chars().filter(|&c| c == '}').count();
            let open_brackets = input_buffer.chars().filter(|&c| c == '[').count();
            let close_brackets = input_buffer.chars().filter(|&c| c == ']').count();

            if open_braces > close_braces || open_brackets > close_brackets {
                continue;
            }

            let input = input_buffer.trim().to_string();
            input_buffer.clear();
            if input.is_empty() { continue; }

            let mut parser = parser::Parser::new(&input);
            match parser.parse() {
                Ok(mut ast) => {
                    let mut is_invalid = false;
                    if ast.len() == 1 {
                        if let ast::Statement::Expression(ast::Expr::Identifier(ref name)) = ast[0] {
                            if !compiler.has_var(name) { is_invalid = true; }
                        }
                    }

                    if is_invalid {
                        println!("invalid command, type help for guide");
                        continue;
                    }

                    for stmt in &mut ast { if let ast::Statement::Expression(expr) = stmt { *stmt = ast::Statement::Print { value: expr.clone() }; } }
                    let start_pc = compiler.bytecode.len();
                    match compiler.compile_line(ast) {
                        Ok(_) => {
                            if let Err(e) = vm.execute_from(&compiler.bytecode, start_pc) {
                                eprintln!("🚨 Runtime Error: {} ", e); compiler.bytecode.truncate(start_pc);
                            } else { compiler.bytecode.pop(); }
                        }
                        Err(e) => { eprintln!("🚨 Compilation Error: {}", e); compiler.bytecode.truncate(start_pc); }
                    }
                }
                Err(e) => {
                    if !input.contains(' ') && !input.contains('(') && !input.contains('=') && !input.contains('{') {
                        println!("invalid command, type help for guide");
                    } else {
                        eprintln!("🚨 Syntax Error:\n{} ", e);
                    }
                }
            }
        }
    }
}