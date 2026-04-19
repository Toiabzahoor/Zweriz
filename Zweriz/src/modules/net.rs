use super::{Arg, Ret};
use std::process::Command;
use std::net::{TcpStream, TcpListener};
use std::io::{Read, Write};

pub fn dispatch(func_id: usize, args: &[Arg]) -> Result<Ret, String> {
    match func_id {
        30 => {
            let url = if let Arg::String(s) = &args[0] { s } else { return Err("http_get needs url".to_string()); };
            let output = Command::new("curl").arg("-s").arg("-L").arg(url).output().map_err(|_| "Network Error: Failed to execute curl".to_string())?;
            Ok(Ret::String(String::from_utf8_lossy(&output.stdout).into_owned()))
        }
        31 => {
            let url = if let Arg::String(s) = &args[0] { s } else { return Err("http_post needs url".to_string()); };
            let body = if let Arg::String(s) = &args[1] { s } else { return Err("http_post needs body string".to_string()); };
            let output = Command::new("curl").arg("-s").arg("-X").arg("POST").arg("-d").arg(body).arg(url).output().map_err(|_| "Network Error: Failed to execute curl".to_string())?;
            Ok(Ret::String(String::from_utf8_lossy(&output.stdout).into_owned()))
        }
        32 => {
            let addr = if let Arg::String(s) = &args[0] { s } else { return Err("tcp_send needs address (e.g. 127.0.0.1:8080)".to_string()); };
            let data = if let Arg::String(s) = &args[1] { s } else { return Err("tcp_send needs data string".to_string()); };
            let mut stream = TcpStream::connect(addr).map_err(|e| format!("TCP Connect Error: {}", e))?;
            stream.write_all(data.as_bytes()).map_err(|e| format!("TCP Write Error: {}", e))?;
            Ok(Ret::Void)
        }
        33 => {
            let addr = if let Arg::String(s) = &args[0] { s } else { return Err("tcp_listen needs address (e.g. 0.0.0.0:8080)".to_string()); };
            let listener = TcpListener::bind(addr).map_err(|e| format!("TCP Bind Error: {}", e))?;
            let (mut stream, _) = listener.accept().map_err(|e| format!("TCP Accept Error: {}", e))?;
            let mut buffer = String::new();
            stream.read_to_string(&mut buffer).map_err(|e| format!("TCP Read Error: {}", e))?;
            Ok(Ret::String(buffer))
        }
        _ => Err(format!("Invalid Net function ID: {}", func_id))
    }
}