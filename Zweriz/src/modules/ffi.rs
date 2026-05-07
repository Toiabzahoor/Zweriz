use super::{Arg, Ret};
use std::collections::HashMap;
use std::sync::Mutex;
use libloading::{Library, Symbol};
use lazy_static::lazy_static;

lazy_static! {
    static ref LIBRARIES: Mutex<HashMap<usize, Library>> = Mutex::new(HashMap::new());
    static ref NEXT_ID: Mutex<usize> = Mutex::new(1);
}

pub fn dispatch(func_id: usize, args: &[Arg]) -> Result<Ret, String> {
    match func_id {
        80 => { // ffi.load("path/to/lib.so")
            let path = if let Arg::String(s) = &args[0] { s } else { return Err("ffi.load needs path".to_string()); };
            unsafe {
                let lib = Library::new(path).map_err(|e| format!("FFI Error: {}", e))?;
                let mut libs = LIBRARIES.lock().unwrap();
                let mut id = NEXT_ID.lock().unwrap();
                let current_id = *id;
                libs.insert(current_id, lib);
                *id += 1;
                Ok(Ret::Number(current_id as f64))
            }
        }
        81 => { // ffi.call(handle, "func_name", arg1, arg2...)
            let lib_id = if let Arg::Number(n) = args[0] { n as usize } else { return Err("ffi.call needs lib handle".to_string()); };
            let func_name = if let Arg::String(s) = &args[1] { s } else { return Err("ffi.call needs func name".to_string()); };
            
            let libs = LIBRARIES.lock().unwrap();
            let lib = libs.get(&lib_id).ok_or("FFI Error: Invalid library handle".to_string())?;
            
            unsafe {
                let arg_count = args.len().saturating_sub(2);
                let mut c_args = [0.0; 4];
                
                for i in 0..arg_count.min(4) {
                    if let Arg::Number(n) = &args[i + 2] {
                        c_args[i] = *n;
                    } else {
                        return Err("FFI Error: ffi.call currently only supports numeric arguments".to_string());
                    }
                }
                
                match arg_count {
                    0 => {
                        let func: Symbol<unsafe extern "C" fn() -> f64> = lib.get(func_name.as_bytes()).map_err(|e| format!("FFI Error: {}", e))?;
                        Ok(Ret::Number(func()))
                    }
                    1 => {
                        let func: Symbol<unsafe extern "C" fn(f64) -> f64> = lib.get(func_name.as_bytes()).map_err(|e| format!("FFI Error: {}", e))?;
                        Ok(Ret::Number(func(c_args[0])))
                    }
                    2 => {
                        let func: Symbol<unsafe extern "C" fn(f64, f64) -> f64> = lib.get(func_name.as_bytes()).map_err(|e| format!("FFI Error: {}", e))?;
                        Ok(Ret::Number(func(c_args[0], c_args[1])))
                    }
                    3 => {
                        let func: Symbol<unsafe extern "C" fn(f64, f64, f64) -> f64> = lib.get(func_name.as_bytes()).map_err(|e| format!("FFI Error: {}", e))?;
                        Ok(Ret::Number(func(c_args[0], c_args[1], c_args[2])))
                    }
                    4 => {
                        let func: Symbol<unsafe extern "C" fn(f64, f64, f64, f64) -> f64> = lib.get(func_name.as_bytes()).map_err(|e| format!("FFI Error: {}", e))?;
                        Ok(Ret::Number(func(c_args[0], c_args[1], c_args[2], c_args[3])))
                    }
                    _ => Err("FFI Error: Too many arguments (max 4)".to_string())
                }
            }
        }
        _ => Err(format!("Invalid FFI function ID: {}", func_id))
    }
}