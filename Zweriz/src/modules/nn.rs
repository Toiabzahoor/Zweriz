use std::f64::consts::PI;
use rand::Rng;
use super::{Arg, Ret};

pub fn dispatch(func_id: usize, args: &[Arg]) -> Result<Ret, String> {
    match func_id {
        40 => {
            let (data, cols) = if let Arg::Array { data, cols } = &args[0] { (data, *cols) } else { return Err("nn.relu needs array".to_string()); };
            let mut result = Vec::with_capacity(data.len());
            for &val in data {
                result.push(if val > 0.0 { val } else { 0.0 });
            }
            Ok(Ret::Array { data: result, cols })
        }
        41 => {
            let (data, cols) = if let Arg::Array { data, cols } = &args[0] { (data, *cols) } else { return Err("nn.sigmoid needs array".to_string()); };
            let mut result = Vec::with_capacity(data.len());
            for &val in data {
                result.push(1.0 / (1.0 + (-val).exp()));
            }
            Ok(Ret::Array { data: result, cols })
        }
        42 => {
            let (data, cols) = if let Arg::Array { data, cols } = &args[0] { (data, *cols) } else { return Err("nn.softmax needs array".to_string()); };
            if data.is_empty() { return Ok(Ret::Array { data: vec![], cols }); }

            let mut max_val = data[0];
            for &val in data { if val > max_val { max_val = val; } }

            let mut exp_sums = 0.0;
            let mut exps = Vec::with_capacity(data.len());
            for &val in data {
                let e = (val - max_val).exp();
                exps.push(e);
                exp_sums += e;
            }

            for val in &mut exps { *val /= exp_sums; }
            Ok(Ret::Array { data: exps, cols })
        }
        47 => {
            let (data, cols) = if let Arg::Array { data, cols } = &args[0] { (data, *cols) } else { return Err("nn.tanh needs array".to_string()); };
            let mut result = Vec::with_capacity(data.len());
            for &val in data {
                result.push(val.tanh());
            }
            Ok(Ret::Array { data: result, cols })
        }
        48 => {
            let (data, cols) = if let Arg::Array { data, cols } = &args[0] { (data, *cols) } else { return Err("nn.leaky_relu needs array".to_string()); };
            let alpha = if args.len() > 1 {
                if let Arg::Number(n) = args[1] { n } else { return Err("nn.leaky_relu alpha must be a number".to_string()); }
            } else { 0.01 };

            let mut result = Vec::with_capacity(data.len());
            for &val in data {
                result.push(if val > 0.0 { val } else { val * alpha });
            }
            Ok(Ret::Array { data: result, cols })
        }
        49 => {
            let (data, cols) = if let Arg::Array { data, cols } = &args[0] { (data, *cols) } else { return Err("nn.gelu needs array".to_string()); };
            let mut result = Vec::with_capacity(data.len());
            let c1 = (2.0 / PI).sqrt();
            for &val in data {
                let inner = c1 * (val + 0.044715 * val.powi(3));
                result.push(0.5 * val * (1.0 + inner.tanh()));
            }
            Ok(Ret::Array { data: result, cols })
        }
        50 => {
            let (data, cols) = if let Arg::Array { data, cols } = &args[0] { (data, *cols) } else { return Err("nn.dropout needs array".to_string()); };
            let p = if args.len() > 1 {
                if let Arg::Number(n) = args[1] { n } else { return Err("nn.dropout probability must be a number".to_string()); }
            } else { 0.5 };

            let mut result = Vec::with_capacity(data.len());
            let mut rng = rand::thread_rng();
            let scale = 1.0 / (1.0 - p);

            for &val in data {
                if rng.r#gen::<f64>() < p {
                    result.push(0.0);
                } else {
                    result.push(val * scale);
                }
            }
            Ok(Ret::Array { data: result, cols })
        }
        51 => {
            let (data, cols) = if let Arg::Array { data, cols } = &args[0] { (data, *cols) } else { return Err("nn.swish needs array".to_string()); };
            let mut result = Vec::with_capacity(data.len());
            for &val in data {
                let sig = 1.0 / (1.0 + (-val).exp());
                result.push(val * sig);
            }
            Ok(Ret::Array { data: result, cols })
        }
        52 => {
            let (data, cols) = if let Arg::Array { data, cols } = &args[0] { (data, *cols) } else { return Err("nn.softplus needs array".to_string()); };
            let mut result = Vec::with_capacity(data.len());
            for &val in data {
                result.push((1.0 + val.exp()).ln());
            }
            Ok(Ret::Array { data: result, cols })
        }
        _ => Err(format!("Invalid NN function ID: {}", func_id))
    }
}