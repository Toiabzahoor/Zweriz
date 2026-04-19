pub mod core;
pub mod math;
pub mod net;
pub mod nn;

#[derive(Clone, Debug)]
pub enum Arg {
    Number(f64),
    String(String),
    Array { data: Vec<f64>, cols: usize },
}

#[derive(Clone, Debug)]
pub enum Ret {
    Number(f64),
    String(String),
    Array { data: Vec<f64>, cols: usize },
    Void,
}

pub fn dispatch(func_id: usize, args: &[Arg]) -> Result<Ret, String> {
    match func_id {

        0..=19 | 60..=69 => core::dispatch(func_id, args),

        20..=29 => math::dispatch(func_id, args),

        30..=39 => net::dispatch(func_id, args),

        40..=59 => nn::dispatch(func_id, args),
        _ => Err(format!("Unknown native function ID: {}", func_id))
    }
}