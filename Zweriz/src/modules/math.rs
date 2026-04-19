use super::{Arg, Ret};

pub fn dispatch(func_id: usize, args: &[Arg]) -> Result<Ret, String> {
    match func_id {
        20 => {
            let (data, original_cols) = if let Arg::Array { data, cols } = &args[0] { (data, *cols) } else { return Err("transpose needs array".to_string()); };
            let rows = data.len() / original_cols;
            let mut t_data = vec![0.0; data.len()];
            for r in 0..rows {
                for c in 0..original_cols {
                    t_data[c * rows + r] = data[r * original_cols + c];
                }
            }
            Ok(Ret::Array { data: t_data, cols: rows })
        }
        21 => {
            let (data, _) = if let Arg::Array { data, cols } = &args[0] { (data, *cols) } else { return Err("trapz needs array".to_string()); };
            let dx = if let Arg::Number(n) = args[1] { n } else { return Err("trapz needs dx number".to_string()); };
            if data.len() < 2 { return Ok(Ret::Number(0.0)); }
            let mut sum = 0.0;
            for i in 0..data.len() - 1 { sum += (data[i] + data[i+1]) / 2.0; }
            Ok(Ret::Number(sum * dx))
        }
        22 => {
            let (data, _) = if let Arg::Array { data, cols } = &args[0] { (data, *cols) } else { return Err("gradient needs array".to_string()); };
            let dx = if let Arg::Number(n) = args[1] { n } else { return Err("gradient needs dx number".to_string()); };
            let n = data.len();
            if n < 2 { return Ok(Ret::Array { data: vec![0.0; n], cols: 1 }); }
            let mut grad = vec![0.0; n];
            grad[0] = (data[1] - data[0]) / dx;
            for i in 1..n-1 { grad[i] = (data[i+1] - data[i-1]) / (2.0 * dx); }
            grad[n-1] = (data[n-1] - data[n-2]) / dx;
            Ok(Ret::Array { data: grad, cols: 1 })
        }
        _ => Err("Invalid Math function".to_string())
    }
}