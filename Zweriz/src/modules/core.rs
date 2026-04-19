use std::fs;
use std::io::Write;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use rand::Rng;
use memmap2::MmapOptions;
use super::{Arg, Ret};

pub fn dispatch(func_id: usize, args: &[Arg]) -> Result<Ret, String> {
    match func_id {
        0 => {
            let mut rng = rand::thread_rng();
            Ok(Ret::Number(rng.r#gen::<f64>()))
        }
        1 => {
            let mut rng = rand::thread_rng();
            if args.len() >= 4 {
                let rows = if let Arg::Number(n) = args[0] { n as usize } else { return Err("random.uniform: needs rows".to_string()); };
                let cols = if let Arg::Number(n) = args[1] { n as usize } else { return Err("random.uniform: needs cols".to_string()); };
                let min = if let Arg::Number(n) = args[2] { n } else { return Err("random.uniform: needs min".to_string()); };
                let max = if let Arg::Number(n) = args[3] { n } else { return Err("random.uniform: needs max".to_string()); };

                let size = rows * cols;
                let mut data = Vec::with_capacity(size);
                for _ in 0..size { data.push(rng.gen_range(min..=max)); }
                Ok(Ret::Array { data, cols })
            } else {
                let size = if let Arg::Number(n) = args[0] { n as usize } else { return Err("random.uniform: needs size".to_string()); };
                let mut data = Vec::with_capacity(size);
                for _ in 0..size { data.push(rng.r#gen::<f64>()); }
                Ok(Ret::Array { data, cols: 1 })
            }
        }
        2 => {
            let min = if let Arg::Number(n) = args[0] { n as i64 } else { return Err("random.randint needs min".to_string()); };
            let max = if let Arg::Number(n) = args[1] { n as i64 } else { return Err("random.randint needs max".to_string()); };
            let mut rng = rand::thread_rng();
            Ok(Ret::Number(rng.gen_range(min..=max) as f64))
        }
        3 => {
            if args.is_empty() {
                let mut input = String::new();
                std::io::stdin().read_line(&mut input).map_err(|_| "I/O Error: Cannot read stdin".to_string())?;
                return Ok(Ret::String(input.trim().to_string()));
            }
            let path = if let Arg::String(s) = &args[0] { s } else { return Err("io.read needs path".to_string()); };
            match fs::read_to_string(path) {
                Ok(content) => Ok(Ret::String(content)),
                Err(e) => Err(format!("I/O Error: Cannot read {}: {}", path, e))
            }
        }
        4 => {
            let path = if let Arg::String(s) = &args[0] { s } else { return Err("io.write needs path".to_string()); };
            let content = if let Arg::String(s) = &args[1] { s } else { return Err("io.write needs content".to_string()); };
            if fs::write(path, content).is_err() { return Err("I/O Error: Cannot write file".to_string()); }
            Ok(Ret::Void)
        }
        5 => {
            let path = if let Arg::String(s) = &args[0] { s } else { return Err("io.append needs path".to_string()); };
            let content = if let Arg::String(s) = &args[1] { s } else { return Err("io.append needs content".to_string()); };
            let mut file = fs::OpenOptions::new().create(true).append(true).open(path).map_err(|_| "I/O Error: Cannot open for append".to_string())?;
            file.write_all(content.as_bytes()).map_err(|_| "I/O Error: Cannot append".to_string())?;
            Ok(Ret::Void)
        }
        6 => {
            let path = if let Arg::String(s) = &args[0] { s } else { return Err("io.exists needs path".to_string()); };
            Ok(Ret::Number(if fs::metadata(path).is_ok() { 1.0 } else { 0.0 }))
        }
        7 => {
            let path = if let Arg::String(s) = &args[0] { s } else { return Err("mmap.load_f64 needs path".to_string()); };
            let file = fs::File::open(path).map_err(|_| "Mmap Error: File not found".to_string())?;
            let mmap = unsafe { MmapOptions::new().map(&file).map_err(|_| "Mmap Error: Failed to map memory".to_string())? };
            let float_count = mmap.len() / 8;
            let mut data = Vec::with_capacity(float_count);
            unsafe { let ptr = mmap.as_ptr() as *const f64; for i in 0..float_count { data.push(*ptr.add(i)); } }
            Ok(Ret::Array { data, cols: 1 })
        }
        8 => {
            let code = if let Arg::Number(n) = args[0] { n as i32 } else { 0 };
            std::process::exit(code);
        }
        9 => {
            let var = if let Arg::String(s) = &args[0] { s } else { return Err("os.env needs var name".to_string()); };
            let val = std::env::var(var).unwrap_or_else(|_| "".to_string());
            Ok(Ret::String(val))
        }
        10 => {
            let cmd_str = if let Arg::String(s) = &args[0] { s } else { return Err("os.cmd needs command".to_string()); };
            let output = Command::new("sh").arg("-c").arg(cmd_str).output().map_err(|_| "OS Error: Failed to execute command".to_string())?;
            Ok(Ret::String(String::from_utf8_lossy(&output.stdout).into_owned()))
        }
        11 => {
            let secs = if let Arg::Number(n) = args[0] { n } else { return Err("time.sleep needs seconds".to_string()); };
            std::thread::sleep(std::time::Duration::from_secs_f64(secs));
            Ok(Ret::Void)
        }
        12 => {
            let start = SystemTime::now();
            let since_the_epoch = start.duration_since(UNIX_EPOCH).expect("Time went backwards");
            Ok(Ret::Number(since_the_epoch.as_secs_f64()))
        }
        13 => {
            let x = if let Arg::Number(n) = args[0] { n } else { return Err("math.floor needs number".to_string()); };
            Ok(Ret::Number(x.floor()))
        }
        14 => {
            let x = if let Arg::Number(n) = args[0] { n } else { return Err("math.ceil needs number".to_string()); };
            Ok(Ret::Number(x.ceil()))
        }
        15 => {
            let x = if let Arg::Number(n) = args[0] { n } else { return Err("math.round needs number".to_string()); };
            Ok(Ret::Number(x.round()))
        }
        16 => {
            let base = if let Arg::Number(n) = args[0] { n } else { return Err("math.pow needs base".to_string()); };
            let exp = if let Arg::Number(n) = args[1] { n } else { return Err("math.pow needs exp".to_string()); };
            Ok(Ret::Number(base.powf(exp)))
        }
        17 => {
            let s = if let Arg::String(s) = &args[0] { s } else { return Err("string.len needs string".to_string()); };
            Ok(Ret::Number(s.len() as f64))
        }
        18 => {
            let s = if let Arg::String(s) = &args[0] { s } else { return Err("string.parse_num needs string".to_string()); };
            let parsed = s.trim().parse::<f64>().map_err(|_| format!("Parse Error: '{}' is not a number", s))?;
            Ok(Ret::Number(parsed))
        }
        19 => {
            let rows = if let Some(Arg::Number(n)) = args.get(0) { *n as usize } else { 1 };
            let cols = if let Some(Arg::Number(n)) = args.get(1) { *n as usize } else { 1 };
            let mean = if let Some(Arg::Number(n)) = args.get(2) { *n } else { 0.0 };
            let std_dev = if let Some(Arg::Number(n)) = args.get(3) { *n } else { 1.0 };

            let size = rows * cols;
            let mut data = Vec::with_capacity(size);
            let mut rng = rand::thread_rng();

            for _ in 0..size {
                let u1: f64 = rng.gen_range(1e-15..=1.0);
                let u2: f64 = rng.r#gen();
                let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                data.push(mean + z0 * std_dev);
            }
            Ok(Ret::Array { data, cols })
        }
        60 => {
            let path = if let Arg::String(s) = &args[0] { s } else { return Err("io.delete needs path".to_string()); };
            if fs::remove_file(path).is_err() { return Err("I/O Error: Cannot delete file".to_string()); }
            Ok(Ret::Void)
        }
        61 => {
            let s = if let Arg::String(s) = &args[0] { s } else { return Err("string.to_lower needs string".to_string()); };
            Ok(Ret::String(s.to_lowercase()))
        }
        62 => {
            let s = if let Arg::String(s) = &args[0] { s } else { return Err("string.to_upper needs string".to_string()); };
            Ok(Ret::String(s.to_uppercase()))
        }
        63 => {
            let s = if let Arg::String(s) = &args[0] { s } else { return Err("string.contains needs string".to_string()); };
            let sub = if let Arg::String(s) = &args[1] { s } else { return Err("string.contains needs substring".to_string()); };
            Ok(Ret::Number(if s.contains(sub) { 1.0 } else { 0.0 }))
        }
        64 => {
            if let Arg::Array { data, cols } = &args[0] {
                let rows = if *cols > 0 { data.len() / *cols } else { 0 };
                Ok(Ret::Array { data: vec![rows as f64, *cols as f64], cols: 2 })
            } else {
                Err("array.shape needs array".to_string())
            }
        }
        65 => {
            if let Arg::Array { data, cols } = &args[0] {
                let mut new_data = data.clone();
                if let Arg::Number(n) = args[1] {
                    new_data.push(n);
                    Ok(Ret::Array { data: new_data, cols: *cols })
                } else { Err("array.push needs number as second arg".to_string()) }
            } else { Err("array.push needs array as first arg".to_string()) }
        }
        66 => {
            if let Arg::Array { data, cols } = &args[0] {
                let mut new_data = data.clone();
                new_data.pop();
                Ok(Ret::Array { data: new_data, cols: *cols })
            } else { Err("array.pop needs array".to_string()) }
        }
        67 => {
            if let Arg::Array { data, cols } = &args[0] {
                Ok(Ret::Array { data: data.clone(), cols: *cols })
            } else { Err("array.clone needs array".to_string()) }
        }
        68 => {
            if let (Arg::Array { data: d1, cols: c1 }, Arg::Array { data: d2, cols: _c2 }) = (&args[0], &args[1]) {
                let mut new_data = d1.clone();
                new_data.extend(d2.iter());
                Ok(Ret::Array { data: new_data, cols: *c1 })
            } else { Err("array.concat needs two arrays".to_string()) }
        }
        69 => {
            let val = if let Arg::Number(n) = args[0] { n as u64 } else { return Err("math.popcount needs number".to_string()); };
            Ok(Ret::Number(val.count_ones() as f64))
        }
        70 => {
            let val = if let Arg::Number(n) = args[0] { n as u64 } else { return Err("math.tzcnt needs number".to_string()); };
            Ok(Ret::Number(val.trailing_zeros() as f64))
        }
        71 => {
            let val = if let Arg::Number(n) = args[0] { n as u64 } else { return Err("math.lzcnt needs number".to_string()); };
            Ok(Ret::Number(val.leading_zeros() as f64))
        }
        72 => {
            let a = if let Arg::Number(n) = args[0] { n } else { return Err("math.min needs number".to_string()); };
            let b = if let Arg::Number(n) = args[1] { n } else { return Err("math.min needs number".to_string()); };
            Ok(Ret::Number(a.min(b)))
        }
        73 => {
            let a = if let Arg::Number(n) = args[0] { n } else { return Err("math.max needs number".to_string()); };
            let b = if let Arg::Number(n) = args[1] { n } else { return Err("math.max needs number".to_string()); };
            Ok(Ret::Number(a.max(b)))
        }
        74 => {
            let s = if let Arg::String(s) = &args[0] { s } else { return Err("string.char_code_at needs string".to_string()); };
            let idx = if let Arg::Number(n) = args[1] { n as usize } else { return Err("string.char_code_at needs index".to_string()); };
            if let Some(c) = s.chars().nth(idx) {
                Ok(Ret::Number(c as u32 as f64))
            } else {
                Err("Index out of bounds".to_string())
            }
        }
        75 => {
            let code = if let Arg::Number(n) = args[0] { n as u32 } else { return Err("string.from_char_code needs number".to_string()); };
            if let Some(c) = std::char::from_u32(code) {
                Ok(Ret::String(c.to_string()))
            } else {
                Err("Invalid char code".to_string())
            }
        }
        _ => Err(format!("Invalid Core function ID: {}", func_id))
    }
}