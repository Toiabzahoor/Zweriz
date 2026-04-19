// Zweriz/src/vm.rs

use crate::compiler::Opcode;
use crate::gpu::{Gpu, GpuVal};
use crate::modules::{dispatch, Arg, Ret};
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

pub const ENGINE_AUTHOR: &str = "made by toiabzahoor";
const QNAN: u64 = 0x7FF8000000000000;
const POINTER_MASK: u64 = 0x0000FFFFFFFFFFFF;
const TAG_SHIFT: u64 = 48;
const TAG_STRING: u64 = 1; 
const TAG_ARRAY: u64 = 2; 
const TAG_DICT: u64 = 3;

fn pack_string_ptr(ptr: usize) -> f64 { f64::from_bits(QNAN | (TAG_STRING << TAG_SHIFT) | ((ptr as u64) & POINTER_MASK)) }
fn is_string(val: f64) -> bool { let bits = val.to_bits(); (bits & QNAN) == QNAN && ((bits >> TAG_SHIFT) & 0x7) == TAG_STRING }
fn pack_array_ptr(ptr: usize) -> f64 { f64::from_bits(QNAN | (TAG_ARRAY << TAG_SHIFT) | ((ptr as u64) & POINTER_MASK)) }
fn is_array(val: f64) -> bool { let bits = val.to_bits(); (bits & QNAN) == QNAN && ((bits >> TAG_SHIFT) & 0x7) == TAG_ARRAY }
fn pack_dict_ptr(ptr: usize) -> f64 { f64::from_bits(QNAN | (TAG_DICT << TAG_SHIFT) | ((ptr as u64) & POINTER_MASK)) }
fn is_dict(val: f64) -> bool { let bits = val.to_bits(); (bits & QNAN) == QNAN && ((bits >> TAG_SHIFT) & 0x7) == TAG_DICT }
fn unpack_ptr(val: f64) -> usize { (val.to_bits() & POINTER_MASK) as usize }

#[derive(Debug, Clone)]
enum TapeOp { Add, Sub, Mul, MatMul, Relu, Sigmoid }

#[derive(Debug, Clone)]
struct TapeNode { op: TapeOp, out_ptr: usize, in_ptrs: Vec<usize> }

struct StringObject { data: String, marked: bool, live: bool }
struct DictObject { data: HashMap<String, f64>, marked: bool, live: bool }
struct ArrayMeta { block_size: usize, marked: bool, live: bool, ref_count: usize }
struct CallFrame { return_pc: usize, bp: usize, return_reg: usize }
struct CatchFrame { bp: usize, catch_pc: usize, err_reg: usize }

pub struct Vm {
    pub engine_signature: &'static str,
    memory: Vec<f64>, bp: usize, frames: Vec<CallFrame>, catch_stack: Vec<CatchFrame>,
    heap: Vec<f64>, heap_ptr: usize, free_blocks: Vec<(usize, usize)>,
    string_arena: Vec<StringObject>, free_strings: Vec<usize>,
    dict_arena: Vec<DictObject>, free_dicts: Vec<usize>, 
    array_arena: HashMap<usize, ArrayMeta>,
    gpu_ctx: Gpu, gpu_dirty: HashSet<usize>, fallback_warned: bool,
    gpu_mode: bool,
    tape: Vec<TapeNode>, grad_map: HashMap<usize, usize>, tracked_tensors: HashSet<usize>,
}

impl Vm {
    pub fn new() -> Self {
        Self {
            engine_signature: "Zweriz Core - made by toiabzahoor",
            memory: vec![0.0; 65536], bp: 0, frames: Vec::with_capacity(256), catch_stack: Vec::new(),
            heap: vec![0.0; 128_000_000], heap_ptr: 0, free_blocks: Vec::new(),
            string_arena: Vec::new(), free_strings: Vec::new(),
            dict_arena: Vec::new(), free_dicts: Vec::new(), array_arena: HashMap::new(),
            gpu_ctx: Gpu::new(), gpu_dirty: HashSet::new(), fallback_warned: false,
            gpu_mode: false, tape: Vec::new(), grad_map: HashMap::new(), tracked_tensors: HashSet::new(),
        }
    }

    #[inline] fn retain_ptr(&mut self, ptr: usize) { if let Some(arr) = self.array_arena.get_mut(&ptr) { if arr.live { arr.ref_count += 1; } } }
    
    #[inline] fn release_ptr(&mut self, ptr: usize) { 
        let mut free_grad = None; 
        if let Some(arr) = self.array_arena.get_mut(&ptr) { 
            if arr.live && arr.ref_count > 0 { 
                arr.ref_count -= 1; 
                if arr.ref_count == 0 { 
                    arr.live = false; 
                    self.gpu_ctx.invalidate(ptr); 
                    self.gpu_dirty.remove(&ptr); 
                    self.free_blocks.push((ptr, arr.block_size)); 
                    self.tracked_tensors.remove(&ptr); 
                    if let Some(grad_ptr) = self.grad_map.remove(&ptr) { free_grad = Some(grad_ptr); } 
                } 
            } 
        } 
        if let Some(g) = free_grad { self.release_ptr(g); } 
    }
    
    #[inline] fn retain(&mut self, val: f64) { if is_array(val) { self.retain_ptr(unpack_ptr(val)); } }
    #[inline] fn release(&mut self, val: f64) { if is_array(val) { self.release_ptr(unpack_ptr(val)); } }
    #[inline] fn set_reg(&mut self, abs_idx: usize, val: f64) { let old = self.memory[abs_idx]; if is_array(val) { self.retain_ptr(unpack_ptr(val)); } self.memory[abs_idx] = val; if is_array(old) { self.release_ptr(unpack_ptr(old)); } }
    
    fn tensor_rank(&self, ptr: usize) -> usize { self.heap[ptr] as usize }
    fn tensor_shape(&self, ptr: usize) -> Vec<usize> { let rank = self.tensor_rank(ptr); (0..rank).map(|i| self.heap[ptr + 1 + i] as usize).collect() }
    fn tensor_data_size(&self, ptr: usize) -> usize { self.tensor_shape(ptr).iter().product() }
    fn data_off(&self, ptr: usize) -> usize { ptr + 1 + self.tensor_rank(ptr) }
    fn is_tracked(&self, val: f64) -> bool { if is_array(val) { self.tracked_tensors.contains(&unpack_ptr(val)) } else { false } }
    
    fn track_tensor(&mut self, ptr: usize) {
        self.tracked_tensors.insert(ptr);
        if !self.grad_map.contains_key(&ptr) { 
            let shape = self.tensor_shape(ptr); let grad_ptr = self.alloc(&shape);
            let size = self.tensor_data_size(grad_ptr); let offset = self.data_off(grad_ptr);
            for i in 0..size { self.heap[offset + i] = 0.0; }
            self.grad_map.insert(ptr, grad_ptr);
        }
    }

    fn clear_tape(&mut self) { 
        let old_tape = std::mem::take(&mut self.tape); 
        for node in old_tape { self.release_ptr(node.out_ptr); for p in node.in_ptrs { self.release_ptr(p); } } 
    }

    fn run_backward(&mut self, use_gpu: bool) {
        for i in (0..self.tape.len()).rev() {
            let node = self.tape[i].clone();
            let dz_ptr = *self.grad_map.get(&node.out_ptr).unwrap();
            let dz_off = self.data_off(dz_ptr);
            let dz_size = self.tensor_data_size(dz_ptr);
            
            if use_gpu && self.gpu_ctx.has_device() {
                match node.op {
                    TapeOp::MatMul => {
                        let (a_ptr, b_ptr) = (node.in_ptrs[0], node.in_ptrs[1]);
                        let a_shape = self.tensor_shape(a_ptr);
                        let b_shape = self.tensor_shape(b_ptr);
                        let (m, k, n) = (a_shape[a_shape.len()-2], a_shape[a_shape.len()-1], b_shape[b_shape.len()-1]);
                        
                        if self.tracked_tensors.contains(&a_ptr) {
                            let da_ptr = *self.grad_map.get(&a_ptr).unwrap();
                            let da_size = m * k;
                            let src = "extern \"C\" __global__ void zweriz_kernel(const unsigned long long* in, unsigned long long* out, const unsigned long long* shapes, size_t size) {
                                size_t idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= size) return;
                                const float* dZ = (const float*)in[0];
                                const float* B = (const float*)in[1];
                                float* dA = (float*)out[0];
                                size_t n = shapes[0], k = shapes[1];
                                size_t i = idx / k; size_t _k = idx % k;
                                float sum = 0.0f;
                                for (size_t j = 0; j < n; j++) { sum += dZ[i * n + j] * B[_k * n + j]; }
                                dA[idx] += sum;
                            }";
                            let inputs = vec![
                                GpuVal::Host { ptr: dz_ptr, offset: dz_off, size: dz_size, shape_col: n },
                                GpuVal::Host { ptr: b_ptr, offset: self.data_off(b_ptr), size: k * n, shape_col: k }
                            ];
                            let outputs = vec![(da_ptr, self.data_off(da_ptr), da_size)];
                            self.gpu_ctx.run_kernel(src, &inputs, &outputs, &self.heap, da_size);
                        }
                        
                        if self.tracked_tensors.contains(&b_ptr) {
                            let db_ptr = *self.grad_map.get(&b_ptr).unwrap();
                            let db_size = k * n;
                            let src = "extern \"C\" __global__ void zweriz_kernel(const unsigned long long* in, unsigned long long* out, const unsigned long long* shapes, size_t size) {
                                size_t idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= size) return;
                                const float* dZ = (const float*)in[0];
                                const float* A = (const float*)in[1];
                                float* dB = (float*)out[0];
                                size_t m = shapes[0], k = shapes[1];
                                size_t n = size / k;
                                size_t _k = idx / n; size_t j = idx % n;
                                float sum = 0.0f;
                                for (size_t i = 0; i < m; i++) { sum += A[i * k + _k] * dZ[i * n + j]; }
                                dB[idx] += sum;
                            }";
                            let inputs = vec![
                                GpuVal::Host { ptr: dz_ptr, offset: dz_off, size: dz_size, shape_col: m },
                                GpuVal::Host { ptr: a_ptr, offset: self.data_off(a_ptr), size: m * k, shape_col: k }
                            ];
                            let outputs = vec![(db_ptr, self.data_off(db_ptr), db_size)];
                            self.gpu_ctx.run_kernel(src, &inputs, &outputs, &self.heap, db_size);
                        }
                    }
                    TapeOp::Add | TapeOp::Sub | TapeOp::Mul => {
                        let (a_ptr, b_ptr) = (node.in_ptrs[0], node.in_ptrs[1]);
                        let op_str = match node.op { TapeOp::Add => "+=", TapeOp::Sub => "-=", TapeOp::Mul => "+=", _ => "" };
                        
                        if self.tracked_tensors.contains(&a_ptr) {
                            let da_ptr = *self.grad_map.get(&a_ptr).unwrap();
                            let mut src = "extern \"C\" __global__ void zweriz_kernel(const unsigned long long* in, unsigned long long* out, const unsigned long long* shapes, size_t size) {\nsize_t idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= size) return;\nconst float* dZ = (const float*)in[0];\n".to_string();
                            let mut inputs = vec![GpuVal::Host { ptr: dz_ptr, offset: dz_off, size: dz_size, shape_col: 1 }];
                            
                            if let TapeOp::Mul = node.op {
                                src.push_str("const float* B = (const float*)in[1];\n((float*)out[0])[idx] += dZ[idx] * B[idx];\n}");
                                inputs.push(GpuVal::Host { ptr: b_ptr, offset: self.data_off(b_ptr), size: dz_size, shape_col: 1 });
                            } else {
                                src.push_str(&format!("((float*)out[0])[idx] {} dZ[idx];\n}}", op_str));
                            }
                            let outputs = vec![(da_ptr, self.data_off(da_ptr), dz_size)];
                            self.gpu_ctx.run_kernel(&src, &inputs, &outputs, &self.heap, dz_size);
                        }
                        
                        if self.tracked_tensors.contains(&b_ptr) {
                            let db_ptr = *self.grad_map.get(&b_ptr).unwrap();
                            let mut src = "extern \"C\" __global__ void zweriz_kernel(const unsigned long long* in, unsigned long long* out, const unsigned long long* shapes, size_t size) {\nsize_t idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= size) return;\nconst float* dZ = (const float*)in[0];\n".to_string();
                            let mut inputs = vec![GpuVal::Host { ptr: dz_ptr, offset: dz_off, size: dz_size, shape_col: 1 }];
                            
                            if let TapeOp::Mul = node.op {
                                src.push_str("const float* A = (const float*)in[1];\n((float*)out[0])[idx] += dZ[idx] * A[idx];\n}");
                                inputs.push(GpuVal::Host { ptr: a_ptr, offset: self.data_off(a_ptr), size: dz_size, shape_col: 1 });
                            } else {
                                let b_op = if let TapeOp::Sub = node.op { "-=" } else { "+=" };
                                src.push_str(&format!("((float*)out[0])[idx] {} dZ[idx];\n}}", b_op));
                            }
                            let outputs = vec![(db_ptr, self.data_off(db_ptr), dz_size)];
                            self.gpu_ctx.run_kernel(&src, &inputs, &outputs, &self.heap, dz_size);
                        }
                    }
                    TapeOp::Relu | TapeOp::Sigmoid => {
                        let a_ptr = node.in_ptrs[0];
                        if self.tracked_tensors.contains(&a_ptr) {
                            let da_ptr = *self.grad_map.get(&a_ptr).unwrap();
                            let mut src = "extern \"C\" __global__ void zweriz_kernel(const unsigned long long* in, unsigned long long* out, const unsigned long long* shapes, size_t size) {\nsize_t idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= size) return;\nconst float* dZ = (const float*)in[0];\nconst float* out_val = (const float*)in[1];\nfloat* dA = (float*)out[0];\n".to_string();
                            
                            if let TapeOp::Relu = node.op {
                                src.push_str("dA[idx] += dZ[idx] * (out_val[idx] > 0.0f ? 1.0f : 0.0f);\n}");
                            } else {
                                src.push_str("float s = out_val[idx];\ndA[idx] += dZ[idx] * s * (1.0f - s);\n}");
                            }
                            
                            let inputs = vec![
                                GpuVal::Host { ptr: dz_ptr, offset: dz_off, size: dz_size, shape_col: 1 },
                                GpuVal::Host { ptr: node.out_ptr, offset: self.data_off(node.out_ptr), size: dz_size, shape_col: 1 }
                            ];
                            let outputs = vec![(da_ptr, self.data_off(da_ptr), dz_size)];
                            self.gpu_ctx.run_kernel(&src, &inputs, &outputs, &self.heap, dz_size);
                        }
                    }
                }
            } else {
                self.sync_cpu(pack_array_ptr(dz_ptr));
                match node.op {
                    TapeOp::MatMul => {
                        let (a_ptr, b_ptr) = (node.in_ptrs[0], node.in_ptrs[1]);
                        let a_shape = self.tensor_shape(a_ptr);
                        let b_shape = self.tensor_shape(b_ptr);
                        let (m, k, n) = (a_shape[a_shape.len()-2], a_shape[a_shape.len()-1], b_shape[b_shape.len()-1]);
                        let a_off = self.data_off(a_ptr);
                        let b_off = self.data_off(b_ptr);
                        
                        if self.tracked_tensors.contains(&a_ptr) {
                            let da_ptr = *self.grad_map.get(&a_ptr).unwrap();
                            self.sync_cpu(pack_array_ptr(da_ptr));
                            let da_off = self.data_off(da_ptr);
                            for i in 0..m {
                                for _k in 0..k {
                                    let mut sum = 0.0;
                                    for j in 0..n { sum += self.heap[dz_off + i * n + j] * self.heap[b_off + _k * n + j]; }
                                    self.heap[da_off + i * k + _k] += sum;
                                }
                            }
                            self.gpu_ctx.invalidate(da_ptr);
                        }
                        if self.tracked_tensors.contains(&b_ptr) {
                            let db_ptr = *self.grad_map.get(&b_ptr).unwrap();
                            self.sync_cpu(pack_array_ptr(db_ptr));
                            let db_off = self.data_off(db_ptr);
                            for _k in 0..k {
                                for j in 0..n {
                                    let mut sum = 0.0;
                                    for i in 0..m { sum += self.heap[a_off + i * k + _k] * self.heap[dz_off + i * n + j]; }
                                    self.heap[db_off + _k * n + j] += sum;
                                }
                            }
                            self.gpu_ctx.invalidate(db_ptr);
                        }
                    }
                    TapeOp::Add => {
                        let (a_ptr, b_ptr) = (node.in_ptrs[0], node.in_ptrs[1]);
                        if self.tracked_tensors.contains(&a_ptr) { let da = *self.grad_map.get(&a_ptr).unwrap(); self.sync_cpu(pack_array_ptr(da)); let off = self.data_off(da); for i in 0..dz_size { self.heap[off + i] += self.heap[dz_off + i]; } self.gpu_ctx.invalidate(da); }
                        if self.tracked_tensors.contains(&b_ptr) { let db = *self.grad_map.get(&b_ptr).unwrap(); self.sync_cpu(pack_array_ptr(db)); let off = self.data_off(db); for i in 0..dz_size { self.heap[off + i] += self.heap[dz_off + i]; } self.gpu_ctx.invalidate(db); }
                    }
                    TapeOp::Sub => {
                        let (a_ptr, b_ptr) = (node.in_ptrs[0], node.in_ptrs[1]);
                        if self.tracked_tensors.contains(&a_ptr) { let da = *self.grad_map.get(&a_ptr).unwrap(); self.sync_cpu(pack_array_ptr(da)); let off = self.data_off(da); for i in 0..dz_size { self.heap[off + i] += self.heap[dz_off + i]; } self.gpu_ctx.invalidate(da); }
                        if self.tracked_tensors.contains(&b_ptr) { let db = *self.grad_map.get(&b_ptr).unwrap(); self.sync_cpu(pack_array_ptr(db)); let off = self.data_off(db); for i in 0..dz_size { self.heap[off + i] -= self.heap[dz_off + i]; } self.gpu_ctx.invalidate(db); }
                    }
                    TapeOp::Mul => {
                        let (a_ptr, b_ptr) = (node.in_ptrs[0], node.in_ptrs[1]);
                        let a_off = self.data_off(a_ptr);
                        let b_off = self.data_off(b_ptr);
                        if self.tracked_tensors.contains(&a_ptr) { let da = *self.grad_map.get(&a_ptr).unwrap(); self.sync_cpu(pack_array_ptr(da)); let off = self.data_off(da); for i in 0..dz_size { self.heap[off + i] += self.heap[dz_off + i] * self.heap[b_off + i]; } self.gpu_ctx.invalidate(da); }
                        if self.tracked_tensors.contains(&b_ptr) { let db = *self.grad_map.get(&b_ptr).unwrap(); self.sync_cpu(pack_array_ptr(db)); let off = self.data_off(db); for i in 0..dz_size { self.heap[off + i] += self.heap[dz_off + i] * self.heap[a_off + i]; } self.gpu_ctx.invalidate(db); }
                    }
                    TapeOp::Relu => {
                        let a_ptr = node.in_ptrs[0];
                        let out_off = self.data_off(node.out_ptr);
                        if self.tracked_tensors.contains(&a_ptr) {
                            let da = *self.grad_map.get(&a_ptr).unwrap(); self.sync_cpu(pack_array_ptr(da)); let off = self.data_off(da);
                            for i in 0..dz_size { self.heap[off + i] += self.heap[dz_off + i] * if self.heap[out_off + i] > 0.0 { 1.0 } else { 0.0 }; }
                            self.gpu_ctx.invalidate(da);
                        }
                    }
                    TapeOp::Sigmoid => {
                        let a_ptr = node.in_ptrs[0];
                        let out_off = self.data_off(node.out_ptr);
                        if self.tracked_tensors.contains(&a_ptr) {
                            let da = *self.grad_map.get(&a_ptr).unwrap(); self.sync_cpu(pack_array_ptr(da)); let off = self.data_off(da);
                            for i in 0..dz_size { let s = self.heap[out_off + i]; self.heap[off + i] += self.heap[dz_off + i] * s * (1.0 - s); }
                            self.gpu_ctx.invalidate(da);
                        }
                    }
                }
            }
        }
    }

    fn sync_cpu(&mut self, val: f64) {
        if is_array(val) { 
            let ptr = unpack_ptr(val); 
            if self.gpu_dirty.contains(&ptr) { 
                let size = self.tensor_data_size(ptr); 
                let offset = self.data_off(ptr); 
                let mut temp = vec![0.0; size]; 
                self.gpu_ctx.pull(ptr, &mut temp); 
                for i in 0..size { self.heap[offset + i] = temp[i]; } 
                self.gpu_dirty.remove(&ptr); 
            } 
        }
    }

    fn alloc(&mut self, shape: &[usize]) -> usize {
        let data_size: usize = shape.iter().product(); 
        let rank = shape.len(); 
        let req_size = 1 + rank + data_size;
        
        let mut found_idx = None; 
        for (i, block) in self.free_blocks.iter().enumerate() { 
            if block.1 >= req_size { found_idx = Some(i); break; } 
        }
        
        if found_idx.is_none() && (self.heap_ptr + req_size > self.heap.len() * 3 / 4) {
            self.run_gc();
            for (i, block) in self.free_blocks.iter().enumerate() { 
                if block.1 >= req_size { found_idx = Some(i); break; } 
            }
        }
        
        let ptr = if let Some(i) = found_idx { 
            let block = self.free_blocks.remove(i); 
            let ptr = block.0; 
            let leftover = block.1 - req_size; 
            if leftover > 0 { self.free_blocks.push((ptr + req_size, leftover)); } 
            ptr 
        } else { 
            let ptr = self.heap_ptr; 
            if self.heap_ptr + req_size > self.heap.len() { 
                let new_size = (self.heap.len() * 2).max(self.heap_ptr + req_size); 
                self.heap.resize(new_size, 0.0); 
            } 
            self.heap_ptr += req_size; 
            ptr 
        };
        
        self.heap[ptr] = rank as f64; 
        for (i, &dim) in shape.iter().enumerate() { self.heap[ptr + 1 + i] = dim as f64; }
        self.array_arena.insert(ptr, ArrayMeta { block_size: req_size, marked: false, live: true, ref_count: 0 }); 
        ptr
    }

    fn stringify(&mut self, val: f64) -> String {
        if is_string(val) { 
            self.string_arena[unpack_ptr(val)].data.clone() 
        } else if is_array(val) { 
            self.sync_cpu(val); 
            let ptr = unpack_ptr(val); 
            let size = self.tensor_data_size(ptr); 
            let offset = self.data_off(ptr); 
            let mut s = String::from("["); 
            for i in 0..size.min(10) { 
                s.push_str(&self.heap[offset + i].to_string()); 
                if i < size - 1 && i < 9 { s.push_str(", "); } 
            } 
            if size > 10 { s.push_str(", ..."); } 
            s.push(']'); 
            s 
        } else { 
            val.to_string() 
        }
    }
    
    fn allocate_string(&mut self, text: String) -> usize { 
        if self.string_arena.len() > 100_000 { self.run_gc(); } 
        if let Some(free_idx) = self.free_strings.pop() { 
            self.string_arena[free_idx] = StringObject { data: text, marked: false, live: true }; 
            free_idx 
        } else { 
            let idx = self.string_arena.len(); 
            self.string_arena.push(StringObject { data: text, marked: false, live: true }); 
            idx 
        } 
    }

    fn run_gc(&mut self) {
        for obj in &mut self.string_arena { obj.marked = false; } 
        for obj in &mut self.dict_arena { obj.marked = false; } 
        for arr in self.array_arena.values_mut() { arr.marked = false; }
        
        let mut worklist = Vec::new();
        
        let active_limit = (self.bp + 256).min(self.memory.len());
        for i in 0..active_limit { 
            let val = self.memory[i]; 
            if is_string(val) { self.string_arena[unpack_ptr(val)].marked = true; } 
            if is_dict(val) { 
                let ptr = unpack_ptr(val);
                if !self.dict_arena[ptr].marked {
                    self.dict_arena[ptr].marked = true;
                    worklist.push(val);
                }
            } 
            if is_array(val) { 
                let ptr = unpack_ptr(val); 
                if let Some(arr) = self.array_arena.get_mut(&ptr) { 
                    if arr.live && !arr.marked { 
                        arr.marked = true;
                        worklist.push(val);
                    } 
                } 
            } 
        }

        while let Some(val) = worklist.pop() {
            if is_dict(val) {
                let dict_ptr = unpack_ptr(val);
                let vals: Vec<f64> = self.dict_arena[dict_ptr].data.values().cloned().collect();
                for v in vals {
                    if is_string(v) { self.string_arena[unpack_ptr(v)].marked = true; }
                    if is_dict(v) {
                        let p = unpack_ptr(v);
                        if !self.dict_arena[p].marked {
                            self.dict_arena[p].marked = true;
                            worklist.push(v);
                        }
                    }
                    if is_array(v) {
                        let p = unpack_ptr(v);
                        if let Some(arr) = self.array_arena.get_mut(&p) {
                            if arr.live && !arr.marked {
                                arr.marked = true;
                                worklist.push(v);
                            }
                        }
                    }
                }
            } else if is_array(val) {
                let ptr = unpack_ptr(val);
                let size = self.tensor_data_size(ptr);
                let offset = self.data_off(ptr);
                for i in 0..size {
                    let v = self.heap[offset + i];
                    if is_string(v) { self.string_arena[unpack_ptr(v)].marked = true; }
                    if is_dict(v) {
                        let p = unpack_ptr(v);
                        if !self.dict_arena[p].marked {
                            self.dict_arena[p].marked = true;
                            worklist.push(v);
                        }
                    }
                    if is_array(v) {
                        let p = unpack_ptr(v);
                        if let Some(arr) = self.array_arena.get_mut(&p) {
                            if arr.live && !arr.marked {
                                arr.marked = true;
                                worklist.push(v);
                            }
                        }
                    }
                }
            }
        }

        let mut grad_ptrs_to_mark = Vec::new();
        for (ptr, grad_ptr) in &self.grad_map {
            if let Some(arr) = self.array_arena.get(ptr) {
                if arr.marked || arr.ref_count > 0 { grad_ptrs_to_mark.push(*grad_ptr); }
            }
        }
        for grad_ptr in grad_ptrs_to_mark {
            if let Some(grad_arr) = self.array_arena.get_mut(&grad_ptr) { grad_arr.marked = true; }
        }
        
        for i in 0..self.string_arena.len() { if self.string_arena[i].live && !self.string_arena[i].marked { self.string_arena[i].data.clear(); self.string_arena[i].live = false; self.free_strings.push(i); } }
        
        for i in 0..self.dict_arena.len() { 
            if self.dict_arena[i].live && !self.dict_arena[i].marked { 
                let vals: Vec<f64> = self.dict_arena[i].data.values().cloned().collect();
                for v in vals {
                    if is_array(v) { self.release(v); }
                }
                self.dict_arena[i].data.clear(); 
                self.dict_arena[i].live = false; 
                self.free_dicts.push(i); 
            } 
        }
        
        let mut dead_ptrs = Vec::new();
        for (&ptr, arr) in self.array_arena.iter_mut() { 
            if arr.live && !arr.marked && arr.ref_count == 0 { 
                arr.live = false; 
                dead_ptrs.push((ptr, arr.block_size));
            } 
        }
        for (ptr, size) in dead_ptrs {
            self.gpu_ctx.invalidate(ptr); 
            self.gpu_dirty.remove(&ptr); 
            self.free_blocks.push((ptr, size)); 
            self.tracked_tensors.remove(&ptr); 
            self.grad_map.remove(&ptr); 
        }
        
        self.free_blocks.sort_by_key(|b| b.0); 
        let mut merged: Vec<(usize, usize)> = Vec::new(); 
        for block in &self.free_blocks { 
            if let Some(last) = merged.last_mut() { 
                if last.0 + last.1 == block.0 { last.1 += block.1; } else { merged.push(*block); } 
            } else { merged.push(*block); } 
        } 
        self.free_blocks = merged;
    }

    fn perform_unary_math(&mut self, val: f64, op: fn(f64) -> f64) -> Result<f64, String> {
        self.sync_cpu(val);
        if is_array(val) { 
            let ptr = unpack_ptr(val); 
            let shape = self.tensor_shape(ptr); 
            let size = self.tensor_data_size(ptr); 
            let offset = self.data_off(ptr); 
            let start = self.alloc(&shape); 
            let dst_offset = self.data_off(start); 
            for i in 0..size { self.heap[dst_offset + i] = op(self.heap[offset + i]); } 
            Ok(pack_array_ptr(start)) 
        } else if is_string(val) { Err("Type Error: Unary op on string".to_string()) } else { Ok(op(val)) }
    }

    fn perform_math(&mut self, left: f64, right: f64, op: fn(f64, f64) -> f64, is_add: bool) -> Result<f64, String> {
        self.sync_cpu(left); self.sync_cpu(right);
        if is_add && (is_string(left) || is_string(right)) { 
            let mut new_str = self.stringify(left); 
            new_str.push_str(&self.stringify(right)); 
            let ptr = self.allocate_string(new_str); 
            return Ok(pack_string_ptr(ptr)); 
        }
        if is_array(left) && is_array(right) { 
            let (l_ptr, r_ptr) = (unpack_ptr(left), unpack_ptr(right)); 
            let shape = self.tensor_shape(l_ptr); 
            let size = self.tensor_data_size(l_ptr); 
            let l_off = self.data_off(l_ptr); 
            let r_off = self.data_off(r_ptr); 
            let start = self.alloc(&shape); 
            let d_off = self.data_off(start); 
            for i in 0..size { self.heap[d_off + i] = op(self.heap[l_off + i], self.heap[r_off + i]); } 
            Ok(pack_array_ptr(start)) 
        } else if is_array(left) && !is_array(right) { 
            let l_ptr = unpack_ptr(left); 
            let shape = self.tensor_shape(l_ptr); 
            let size = self.tensor_data_size(l_ptr); 
            let l_off = self.data_off(l_ptr); 
            let start = self.alloc(&shape); 
            let d_off = self.data_off(start); 
            for i in 0..size { self.heap[d_off + i] = op(self.heap[l_off + i], right); } 
            Ok(pack_array_ptr(start)) 
        } else if !is_array(left) && is_array(right) { 
            let r_ptr = unpack_ptr(right); 
            let shape = self.tensor_shape(r_ptr); 
            let size = self.tensor_data_size(r_ptr); 
            let r_off = self.data_off(r_ptr); 
            let start = self.alloc(&shape); 
            let d_off = self.data_off(start); 
            for i in 0..size { self.heap[d_off + i] = op(left, self.heap[r_off + i]); } 
            Ok(pack_array_ptr(start)) 
        } else { 
            if is_string(left) || is_string(right) { return Err("Type Error: Math op on string".to_string()); } 
            Ok(op(left, right)) 
        }
    }

    fn handle_throw(&mut self, err_val: f64, pc: &mut usize) -> Result<(), String> { 
        let err_str = self.stringify(err_val); 
        if let Some(frame) = self.catch_stack.pop() { 
            self.bp = frame.bp; 
            self.set_reg(self.bp + frame.err_reg, err_val); 
            *pc = frame.catch_pc; Ok(()) 
        } else { Err(err_str) } 
    }

    pub fn execute(&mut self, program: &[Opcode]) -> Result<(), String> { self.execute_from(program, 0) }

    pub fn execute_from(&mut self, program: &[Opcode], start_pc: usize) -> Result<(), String> {
        let mut pc = start_pc;
        macro_rules! throw { ($msg:expr) => { { let ptr = self.allocate_string($msg); self.handle_throw(pack_string_ptr(ptr), &mut pc)?; continue; } }; }
        
        macro_rules! do_math_tracked {
            ($d:expr, $l:expr, $r:expr, $op:expr, $is_add:expr, $tape_op:expr) => {{
                let left = self.memory[self.bp + $l]; 
                let right = self.memory[self.bp + $r];
                match self.perform_math(left, right, $op, $is_add) {
                    Ok(res) => {
                        self.set_reg(self.bp + $d, res);
                        if self.is_tracked(left) || self.is_tracked(right) {
                            if is_array(res) {
                                let out_ptr = unpack_ptr(res); self.track_tensor(out_ptr);
                                let mut in_ptrs = Vec::new(); 
                                if is_array(left) { in_ptrs.push(unpack_ptr(left)); } 
                                if is_array(right) { in_ptrs.push(unpack_ptr(right)); }
                                self.tape.push(TapeNode { op: $tape_op, out_ptr, in_ptrs: in_ptrs.clone() });
                                self.retain_ptr(out_ptr);
                                for &p in &in_ptrs { self.retain_ptr(p); }
                            }
                        }
                    },
                    Err(e) => throw!(e),
                }
            }};
        }

        macro_rules! do_math_untracked {
            ($d:expr, $l:expr, $r:expr, $op:expr, $is_add:expr) => {{
                let left = self.memory[self.bp + $l];
                let right = self.memory[self.bp + $r];
                match self.perform_math(left, right, $op, $is_add) { Ok(res) => self.set_reg(self.bp + $d, res), Err(e) => throw!(e) }
            }};
        }
        
        macro_rules! do_unary_untracked {
            ($d:expr, $s:expr, $op:expr) => {{
                match self.perform_unary_math(self.memory[self.bp + $s], $op) { Ok(res) => self.set_reg(self.bp + $d, res), Err(e) => throw!(e) }
            }};
        }

        while pc < program.len() {
            match &program[pc] {
                Opcode::SetGpuMode(mode) => { self.gpu_mode = *mode; },
                Opcode::RunGC => self.run_gc(),
                Opcode::Halt => break,
                Opcode::Jmp(target) => { pc = *target; continue; },
                Opcode::JmpIfFalse(cond_reg, target) => { let cond = self.memory[self.bp + cond_reg]; if cond == 0.0 { pc = *target; continue; } },
                Opcode::Time(d) => { let val = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64(); self.set_reg(self.bp + d, val); },
                Opcode::LoadConst(d, val) => { self.set_reg(self.bp + d, *val); },
                Opcode::LoadString(d, raw_string) => { let ptr = self.allocate_string(raw_string.clone()); self.set_reg(self.bp + d, pack_string_ptr(ptr)); },
                Opcode::Move(d, s) => { let val = self.memory[self.bp + s]; self.set_reg(self.bp + d, val); },
                Opcode::PushCatch(catch_pc, err_reg) => { self.catch_stack.push(CatchFrame { bp: self.bp, catch_pc: *catch_pc, err_reg: *err_reg }); },
                Opcode::PopCatch => { self.catch_stack.pop(); },
                Opcode::Throw(val_reg) => { let err_val = self.memory[self.bp + val_reg]; self.handle_throw(err_val, &mut pc)?; continue; },
                Opcode::Call(target, dest_reg, arg_start, arg_count) => { 
                    self.frames.push(CallFrame { return_pc: pc, bp: self.bp, return_reg: *dest_reg }); 
                    let new_bp = self.bp + 256; 
                    for i in 0..*arg_count { 
                        let val = self.memory[self.bp + arg_start + i];
                        self.set_reg(new_bp + i, val);
                    } 
                    self.bp = new_bp; pc = *target; continue; 
                },
                Opcode::Return(reg) => { 
                    let ret_val = self.memory[self.bp + reg]; 
                    self.retain(ret_val);
                    for i in 0..256 {
                        let v = self.memory[self.bp + i];
                        if is_array(v) { self.release(v); }
                        self.memory[self.bp + i] = 0.0;
                    }
                    if let Some(frame) = self.frames.pop() { 
                        pc = frame.return_pc; 
                        self.bp = frame.bp; 
                        self.set_reg(self.bp + frame.return_reg, ret_val); 
                        self.release(ret_val);
                    } else { break; } 
                },
                
                Opcode::NativeCall(func_id, dest_reg, arg_start, arg_count) => {
                    if *func_id == 23 { 
                        let target_val = self.memory[self.bp + arg_start];
                        let shift_r = self.memory[self.bp + arg_start + 1] as isize;
                        let shift_c = self.memory[self.bp + arg_start + 2] as isize;
                        
                        if is_array(target_val) {
                            let in_ptr = unpack_ptr(target_val);
                            let shape = self.tensor_shape(in_ptr);
                            let size = self.tensor_data_size(in_ptr);
                            let in_off = self.data_off(in_ptr);
                            
                            let rows = if shape.len() > 1 { shape[shape.len()-2] } else { 1 };
                            let cols = *shape.last().unwrap_or(&1);
                            let grid_size = rows * cols;
                            
                            let start = self.alloc(&shape);
                            let d_off = self.data_off(start);
                            
                            if self.gpu_mode && self.gpu_ctx.has_device() {
                                let src = format!("extern \"C\" __global__ void zweriz_kernel(const unsigned long long* in, unsigned long long* out, const unsigned long long* shapes, size_t size) {{
                                    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= size) return;
                                    const float* A = (const float*)in[0];
                                    float* Z = (float*)out[0];
                                    long rows = {}; long cols = {}; long grid_size = {};
                                    long batch_idx = idx / grid_size;
                                    long local_idx = idx % grid_size;
                                    long r = local_idx / cols; long c = local_idx % cols;
                                    long src_r = r - ({}); long src_c = c - ({});
                                    if (src_r >= 0 && src_r < rows && src_c >= 0 && src_c < cols) {{
                                        Z[idx] = A[batch_idx * grid_size + src_r * cols + src_c];
                                    }} else {{
                                        Z[idx] = 0.0f;
                                    }}
                                }}", rows, cols, grid_size, shift_r, shift_c);
                                
                                let inputs = vec![GpuVal::Host { ptr: in_ptr, offset: in_off, size, shape_col: cols }];
                                let outputs = vec![(start, d_off, size)];
                                self.gpu_ctx.run_kernel(&src, &inputs, &outputs, &self.heap, size);
                            } else {
                                let batches = size / grid_size;
                                for batch in 0..batches {
                                    for r in 0..rows {
                                        for c in 0..cols {
                                            let src_r = (r as isize) - shift_r;
                                            let src_c = (c as isize) - shift_c;
                                            let dst_idx = batch * grid_size + r * cols + c;
                                            if src_r >= 0 && src_r < (rows as isize) && src_c >= 0 && src_c < (cols as isize) {
                                                self.heap[d_off + dst_idx] = self.heap[in_off + batch * grid_size + (src_r as usize) * cols + (src_c as usize)];
                                            } else { self.heap[d_off + dst_idx] = 0.0; }
                                        }
                                    }
                                }
                            }
                            self.set_reg(self.bp + *dest_reg, pack_array_ptr(start));
                        } else { throw!("math.shift2d requires an array.".to_string()); }
                    } else if *func_id == 40 || *func_id == 41 { 
                        let input_val = self.memory[self.bp + arg_start];
                        self.sync_cpu(input_val);
                        if is_array(input_val) {
                            let in_ptr = unpack_ptr(input_val);
                            let shape = self.tensor_shape(in_ptr);
                            let size = self.tensor_data_size(in_ptr);
                            let in_off = self.data_off(in_ptr);
                            let start = self.alloc(&shape);
                            let d_off = self.data_off(start);
                            
                            if self.gpu_mode && self.gpu_ctx.has_device() {
                                let mut src = "extern \"C\" __global__ void zweriz_kernel(const unsigned long long* in, unsigned long long* out, const unsigned long long* shapes, size_t size) {\nsize_t idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= size) return;\nconst float* A = (const float*)in[0];\nfloat* Z = (float*)out[0];\n".to_string();
                                if *func_id == 40 { src.push_str("Z[idx] = A[idx] > 0.0f ? A[idx] : 0.0f;\n}"); } else { src.push_str("Z[idx] = 1.0f / (1.0f + exp(-A[idx]));\n}"); }
                                let inputs = vec![GpuVal::Host { ptr: in_ptr, offset: in_off, size, shape_col: 1 }];
                                let outputs = vec![(start, d_off, size)];
                                self.gpu_ctx.run_kernel(&src, &inputs, &outputs, &self.heap, size);
                            } else {
                                for i in 0..size {
                                    let v = self.heap[in_off + i];
                                    self.heap[d_off + i] = if *func_id == 40 { if v > 0.0 { v } else { 0.0 } } else { 1.0 / (1.0 + (-v).exp()) };
                                }
                            }
                            
                            self.set_reg(self.bp + dest_reg, pack_array_ptr(start));
                            if self.is_tracked(input_val) {
                                self.track_tensor(start);
                                self.tape.push(TapeNode { op: if *func_id == 40 { TapeOp::Relu } else { TapeOp::Sigmoid }, out_ptr: start, in_ptrs: vec![in_ptr] });
                                self.retain_ptr(start);
                                self.retain_ptr(in_ptr);
                            }
                        } else { throw!("NN functions require array.".to_string()); }
                    } else if *func_id == 43 { 
                        let val = self.memory[self.bp + arg_start];
                        if is_array(val) { self.track_tensor(unpack_ptr(val)); }
                    } else if *func_id == 44 { 
                        let val = self.memory[self.bp + arg_start];
                        if is_array(val) {
                            let ptr = unpack_ptr(val);
                            if let Some(&grad_ptr) = self.grad_map.get(&ptr) {
                                let size = self.tensor_data_size(grad_ptr);
                                if self.gpu_mode && self.gpu_ctx.has_device() {
                                    let src = "extern \"C\" __global__ void zweriz_kernel(const unsigned long long* in, unsigned long long* out, const unsigned long long* shapes, size_t size) {
                                        size_t idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= size) return;
                                        ((float*)out[0])[idx] = 1.0f;
                                    }";
                                    let outputs = vec![(grad_ptr, self.data_off(grad_ptr), size)];
                                    self.gpu_ctx.run_kernel(src, &[], &outputs, &self.heap, size);
                                } else {
                                    self.sync_cpu(pack_array_ptr(grad_ptr));
                                    let off = self.data_off(grad_ptr);
                                    for i in 0..size { self.heap[off + i] = 1.0; }
                                    self.gpu_ctx.invalidate(grad_ptr);
                                }
                            }
                            self.run_backward(self.gpu_mode);
                        } else { throw!("Backward requires loss array.".to_string()); }
                    } else if *func_id == 45 { 
                        let lr = self.memory[self.bp + arg_start];
                        if self.gpu_mode && self.gpu_ctx.has_device() {
                            let src = "extern \"C\" __global__ void zweriz_kernel(const unsigned long long* in, unsigned long long* out, const unsigned long long* shapes, size_t size) {
                                size_t idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= size) return;
                                const float lr = ((const float*)in[0])[0];
                                const float* dW = (const float*)in[1];
                                float* W = (float*)out[0];
                                W[idx] -= lr * dW[idx];
                            }";
                            for ptr in &self.tracked_tensors {
                                if let Some(&grad_ptr) = self.grad_map.get(ptr) {
                                    let size = self.tensor_data_size(*ptr);
                                    let w_off = self.data_off(*ptr);
                                    let dw_off = self.data_off(grad_ptr);
                                    let inputs = vec![
                                        GpuVal::Scalar(lr),
                                        GpuVal::Host { ptr: grad_ptr, offset: dw_off, size, shape_col: 1 }
                                    ];
                                    let outputs = vec![(*ptr, w_off, size)];
                                    self.gpu_ctx.run_kernel(src, &inputs, &outputs, &self.heap, size);
                                }
                            }
                        } else {
                            let mut updates = Vec::new();
                            for ptr in &self.tracked_tensors {
                                if let Some(&grad_ptr) = self.grad_map.get(ptr) { updates.push((*ptr, grad_ptr)); }
                            }
                            for (ptr, grad_ptr) in updates {
                                let size = self.tensor_data_size(ptr);
                                self.sync_cpu(pack_array_ptr(ptr));
                                self.sync_cpu(pack_array_ptr(grad_ptr));
                                let w_off = self.data_off(ptr);
                                let dw_off = self.data_off(grad_ptr);
                                for i in 0..size { self.heap[w_off + i] -= lr * self.heap[dw_off + i]; }
                                self.gpu_ctx.invalidate(ptr);
                            }
                        }
                    } else if *func_id == 100 { 
                        let input_val = self.memory[self.bp + arg_start];
                        let turn_val = self.memory[self.bp + arg_start + 1];
                        
                        self.sync_cpu(input_val);
                        if is_array(input_val) {
                            let in_ptr = unpack_ptr(input_val);
                            let shape = self.tensor_shape(in_ptr);
                            let in_off = self.data_off(in_ptr);
                            
                            let num_boards = if shape.len() > 1 { shape[0] } else { 1 };
                            let max_moves = 218;
                            
                            let out_boards_ptr = self.alloc(&[num_boards * max_moves, 64]);
                            let out_boards_off = self.data_off(out_boards_ptr);
                            
                            let out_counts_ptr = self.alloc(&[num_boards, 1]);
                            let out_counts_off = self.data_off(out_counts_ptr);
                            
                            if self.gpu_ctx.has_device() {
                                let cuda_src = format!("{}\n{}", 
                                  include_str!("../cuda/heuristics.cu"), 
                                  include_str!("../cuda/chess.cu")
                                        );
                                let src = cuda_src.as_str();
                                let inputs = vec![
                                    crate::gpu::GpuVal::Host { ptr: in_ptr, offset: in_off, size: num_boards * 64, shape_col: 64 },
                                    crate::gpu::GpuVal::Scalar(turn_val)
                                ];
                                let outputs = vec![
                                    (out_boards_ptr, out_boards_off, num_boards * max_moves * 64),
                                    (out_counts_ptr, out_counts_off, num_boards)
                                ];
                                self.gpu_ctx.run_kernel(src, &inputs, &outputs, &self.heap, num_boards);
                            } 
                            
                            let res_ptr = self.alloc(&[2]);
                            let res_off = self.data_off(res_ptr);
                            self.heap[res_off] = pack_array_ptr(out_boards_ptr);
                            self.heap[res_off + 1] = pack_array_ptr(out_counts_ptr);
                            
                            self.set_reg(self.bp + dest_reg, pack_array_ptr(res_ptr));
                        } else { 
                            throw!("Chess generation requires a batch array of boards.".to_string()); 
                        }
                    } else if *func_id == 101 { 
    let input_val = self.memory[self.bp + arg_start];
    let depth_val = self.memory[self.bp + arg_start + 1];
    let _budget_val = self.memory[self.bp + arg_start + 2]; 
    let turn_val = self.memory[self.bp + arg_start + 3];
    
    let small_delta = self.memory[self.bp + arg_start + 4];
    let high_delta = self.memory[self.bp + arg_start + 5];
    let grace_depth = self.memory[self.bp + arg_start + 6];
    
    self.sync_cpu(input_val);
    if is_array(input_val) {
        let in_ptr = unpack_ptr(input_val);
        let in_off = self.data_off(in_ptr);
        
        let max_queue = 16384; 
        let max_moves = 218;
        
        if self.gpu_ctx.has_device() {
            let q_boards = self.alloc(&[max_queue, 64]);
            let q_state = self.alloc(&[max_queue, 1]);
            let q_scores = self.alloc(&[max_queue, 1]);
            let q_depths = self.alloc(&[max_queue, 1]);
            let q_root_ids = self.alloc(&[max_queue, 1]);
            let q_coords = self.alloc(&[max_queue, 4]);
            let q_grace_ttl = self.alloc(&[max_queue, 1]);
            let root_stats = self.alloc(&[max_queue, 5]);
            
            let e_boards = self.alloc(&[max_queue * max_moves, 64]);
            let e_active = self.alloc(&[max_queue * max_moves, 1]);
            let e_scores = self.alloc(&[max_queue * max_moves, 1]);

            let limits = self.alloc(&[1]);
            let limits_off = self.data_off(limits);
            self.heap[limits_off] = depth_val;
            self.gpu_dirty.insert(limits);

            let thresholds = self.alloc(&[3]);
            let t_off = self.data_off(thresholds);
            self.heap[t_off] = small_delta;
            self.heap[t_off + 1] = high_delta;
            self.heap[t_off + 2] = grace_depth;
            self.gpu_dirty.insert(thresholds);

            let global_best = self.alloc(&[1]);
            let g_off = self.data_off(global_best);
            self.gpu_dirty.insert(global_best);
            
            let cuda_src = format!("{}\n{}", include_str!("../cuda/heuristics.cu"), include_str!("../cuda/chess.cu"));
            let src = cuda_src.as_str();
            
            let inputs_init = vec![
                crate::gpu::GpuVal::Host { ptr: in_ptr, offset: in_off, size: 64, shape_col: 64 },
                crate::gpu::GpuVal::Scalar(turn_val)
            ];
            let outputs_init = vec![
                (q_boards, self.data_off(q_boards), max_queue * 64),
                (q_state, self.data_off(q_state), max_queue),
                (q_root_ids, self.data_off(q_root_ids), max_queue),
                (q_depths, self.data_off(q_depths), max_queue),
                (q_coords, self.data_off(q_coords), max_queue * 4),
                (q_grace_ttl, self.data_off(q_grace_ttl), max_queue),
                (root_stats, self.data_off(root_stats), max_queue * 5)
            ];
            self.gpu_ctx.run_named_kernel(src, "init_search_kernel", &inputs_init, &outputs_init, &self.heap, 1);
            
            let reset_src = "extern \"C\" __global__ void zweriz_kernel(const unsigned long long* in, unsigned long long* out, const unsigned long long* shapes, size_t size) { ((float*)out[0])[0] = -1e38f; }";
            self.gpu_ctx.run_kernel(reset_src, &[], &vec![(global_best, g_off, 1)], &self.heap, 1);

            let inputs_h1 = vec![
                crate::gpu::GpuVal::Host { ptr: q_boards, offset: self.data_off(q_boards), size: max_queue * 64, shape_col: 64 },
                crate::gpu::GpuVal::Host { ptr: q_state, offset: self.data_off(q_state), size: max_queue, shape_col: 1 },
                crate::gpu::GpuVal::Scalar(turn_val)
            ];
            let outputs_h1 = vec![
                (q_scores, self.data_off(q_scores), max_queue),
                (global_best, g_off, 1)
            ];
            self.gpu_ctx.run_named_kernel(src, "eval_kernel", &inputs_h1, &outputs_h1, &self.heap, max_queue);
            
            let max_d = depth_val as usize;
            let mut sim_turn = if turn_val == 0.0 { 1.0 } else { 0.0 };
            
            for d in 1..=max_d {
                let cur_depth_f = d as f64;

                let inputs_exp = vec![
                    crate::gpu::GpuVal::Host { ptr: q_boards, offset: self.data_off(q_boards), size: max_queue * 64, shape_col: 64 },
                    crate::gpu::GpuVal::Host { ptr: q_state, offset: self.data_off(q_state), size: max_queue, shape_col: 1 },
                    crate::gpu::GpuVal::Host { ptr: q_depths, offset: self.data_off(q_depths), size: max_queue, shape_col: 1 },
                    crate::gpu::GpuVal::Scalar(sim_turn),
                    crate::gpu::GpuVal::Scalar(cur_depth_f)
                ];
                let outputs_exp = vec![
                    (e_boards, self.data_off(e_boards), max_queue * max_moves * 64),
                    (e_active, self.data_off(e_active), max_queue * max_moves)
                ];
                self.gpu_ctx.run_named_kernel(src, "expand_kernel", &inputs_exp, &outputs_exp, &self.heap, max_queue);
                
                self.gpu_ctx.run_kernel(reset_src, &[], &vec![(global_best, g_off, 1)], &self.heap, 1);

                let inputs_h2 = vec![
                    crate::gpu::GpuVal::Host { ptr: e_boards, offset: self.data_off(e_boards), size: max_queue * max_moves * 64, shape_col: 64 },
                    crate::gpu::GpuVal::Host { ptr: e_active, offset: self.data_off(e_active), size: max_queue * max_moves, shape_col: 1 },
                    crate::gpu::GpuVal::Scalar(sim_turn)
                ];
                let outputs_h2 = vec![
                    (e_scores, self.data_off(e_scores), max_queue * max_moves),
                    (global_best, g_off, 1)
                ];
                self.gpu_ctx.run_named_kernel(src, "eval_kernel", &inputs_h2, &outputs_h2, &self.heap, max_queue * max_moves);
                
                let inputs_commit = vec![
                    crate::gpu::GpuVal::Host { ptr: e_boards, offset: self.data_off(e_boards), size: max_queue * max_moves * 64, shape_col: 64 },
                    crate::gpu::GpuVal::Host { ptr: e_scores, offset: self.data_off(e_scores), size: max_queue * max_moves, shape_col: 1 },
                    crate::gpu::GpuVal::Host { ptr: e_active, offset: self.data_off(e_active), size: max_queue * max_moves, shape_col: 1 },
                    crate::gpu::GpuVal::Host { ptr: q_depths, offset: self.data_off(q_depths), size: max_queue, shape_col: 1 },
                    crate::gpu::GpuVal::Host { ptr: q_grace_ttl, offset: self.data_off(q_grace_ttl), size: max_queue, shape_col: 1 },
                    crate::gpu::GpuVal::Host { ptr: q_root_ids, offset: self.data_off(q_root_ids), size: max_queue, shape_col: 1 },
                    crate::gpu::GpuVal::Host { ptr: thresholds, offset: self.data_off(thresholds), size: 3, shape_col: 1 },
                    crate::gpu::GpuVal::Host { ptr: q_state, offset: self.data_off(q_state), size: max_queue, shape_col: 1 },
                    crate::gpu::GpuVal::Scalar(turn_val),
                    crate::gpu::GpuVal::Scalar(sim_turn),
                    crate::gpu::GpuVal::Host { ptr: global_best, offset: g_off, size: 1, shape_col: 1 }
                ];
                let outputs_commit = vec![
                    (q_boards, self.data_off(q_boards), max_queue * 64),
                    (q_scores, self.data_off(q_scores), max_queue),
                    (q_depths, self.data_off(q_depths), max_queue),
                    (q_state, self.data_off(q_state), max_queue),
                    (q_grace_ttl, self.data_off(q_grace_ttl), max_queue),
                    (q_root_ids, self.data_off(q_root_ids), max_queue),
                    (root_stats, self.data_off(root_stats), max_queue * 5)
                ];
                self.gpu_ctx.run_named_kernel(src, "commit_kernel", &inputs_commit, &outputs_commit, &self.heap, max_queue);
                
                sim_turn = if sim_turn == 0.0 { 1.0 } else { 0.0 };
            }
            
            self.gpu_dirty.insert(root_stats);
            self.gpu_dirty.insert(q_coords);
            
            self.sync_cpu(pack_array_ptr(root_stats));
            self.sync_cpu(pack_array_ptr(q_coords));
            
            let stats_off = self.data_off(root_stats);
            let coord_off = self.data_off(q_coords);
            
            let mut best_score = -f64::INFINITY;
            let mut best_idx = 0;
            
            for i in 0..35 { 
                let count = self.heap[stats_off + i * 5 + 3];
                if count > 0.0 {
                    let max_val = self.heap[stats_off + i * 5 + 0];
                    let min_val = self.heap[stats_off + i * 5 + 1];
                    let avg_val = self.heap[stats_off + i * 5 + 2] / count;
                    let cum_delta = self.heap[stats_off + i * 5 + 4];
                    let avg_delta = cum_delta / count;
                    
                    let mut weighted_score = (min_val * 0.7) + (avg_val * 0.25) + (max_val * 0.05);
                    
                    if avg_delta < 10.0 { 
                        weighted_score -= 100.0;
                    }

                    if weighted_score > best_score {
                        best_score = weighted_score;
                        best_idx = i;
                    }
                }
            }
            
            let res_ptr = self.alloc(&[4]);
            let res_off = self.data_off(res_ptr);
            self.heap[res_off + 0] = self.heap[coord_off + best_idx * 4 + 0];
            self.heap[res_off + 1] = self.heap[coord_off + best_idx * 4 + 1];
            self.heap[res_off + 2] = self.heap[coord_off + best_idx * 4 + 2];
            self.heap[res_off + 3] = self.heap[coord_off + best_idx * 4 + 3];
            
            self.set_reg(self.bp + *dest_reg, pack_array_ptr(res_ptr));
            
        } else {
            throw!("GPU Priority Search requires active CUDA device.".to_string());
        }
    } else {
        throw!("Search requires a board array.".to_string());
    }
} else if *func_id == 46 { 
                        if self.gpu_mode && self.gpu_ctx.has_device() {
                            let src = "extern \"C\" __global__ void zweriz_kernel(const unsigned long long* in, unsigned long long* out, const unsigned long long* shapes, size_t size) {
                                size_t idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= size) return;
                                ((float*)out[0])[idx] = 0.0f;
                            }";
                            for &grad_ptr in self.grad_map.values() {
                                let size = self.tensor_data_size(grad_ptr);
                                let outputs = vec![(grad_ptr, self.data_off(grad_ptr), size)];
                                self.gpu_ctx.run_kernel(src, &[], &outputs, &self.heap, size);
                            }
                        } else {
                            let grad_ptrs: Vec<usize> = self.grad_map.values().cloned().collect();
                            for grad_ptr in grad_ptrs {
                                let size = self.tensor_data_size(grad_ptr);
                                self.sync_cpu(pack_array_ptr(grad_ptr));
                                let off = self.data_off(grad_ptr);
                                for i in 0..size { self.heap[off + i] = 0.0; }
                                self.gpu_ctx.invalidate(grad_ptr);
                            }
                        }
                        self.clear_tape(); 
                    } else {
                        let mut native_args = Vec::with_capacity(*arg_count);
                        for i in 0..*arg_count { 
                            let val = self.memory[self.bp + arg_start + i]; 
                            self.sync_cpu(val); 
                            if is_string(val) { 
                                native_args.push(Arg::String(self.stringify(val))); 
                            } else if is_array(val) { 
                                let ptr = unpack_ptr(val); 
                                let size = self.tensor_data_size(ptr); 
                                let shape = self.tensor_shape(ptr); 
                                let offset = self.data_off(ptr); 
                                let cols = *shape.last().unwrap_or(&1); 
                                let mut data = Vec::with_capacity(size); 
                                for j in 0..size { data.push(self.heap[offset + j]); } 
                                native_args.push(Arg::Array { data, cols }); 
                            } else { native_args.push(Arg::Number(val)); } 
                        }
                        match dispatch(*func_id, &native_args) { 
                            Ok(Ret::Number(n)) => { self.set_reg(self.bp + dest_reg, n); },
                            Ok(Ret::String(s)) => { let ptr = self.allocate_string(s); self.set_reg(self.bp + dest_reg, pack_string_ptr(ptr)); },
                            Ok(Ret::Array { data, cols }) => { 
                                let size = data.len(); 
                                let rows = if cols > 0 { size / cols } else { 0 }; 
                                let shape = vec![rows, cols]; 
                                let start = self.alloc(&shape); 
                                let offset = self.data_off(start); 
                                for (i, val) in data.into_iter().enumerate() { self.heap[offset + i] = val; } 
                                self.set_reg(self.bp + dest_reg, pack_array_ptr(start)); 
                            },
                            Ok(Ret::Void) => { self.set_reg(self.bp + dest_reg, 0.0); },
                            Err(e) => throw!(e),
                        }
                    }
                }

                Opcode::Add(d, l, r) => do_math_tracked!(*d, *l, *r, |a, b| a + b, true, TapeOp::Add),
                Opcode::Sub(d, l, r) => do_math_tracked!(*d, *l, *r, |a, b| a - b, false, TapeOp::Sub),
                Opcode::Mul(d, l, r) => do_math_tracked!(*d, *l, *r, |a, b| a * b, false, TapeOp::Mul),
                
                Opcode::Div(d, l, r) => do_math_untracked!(*d, *l, *r, |a, b| a / b, false),
                Opcode::Mod(d, l, r) => do_math_untracked!(*d, *l, *r, |a, b| a % b, false),
                Opcode::Pow(d, l, r) => do_math_untracked!(*d, *l, *r, |a, b| a.powf(b), false),
                Opcode::BitXor(d, l, r) => do_math_untracked!(*d, *l, *r, |a, b| (a as i64 ^ b as i64) as f64, false),
                Opcode::BitAnd(d, l, r) => do_math_untracked!(*d, *l, *r, |a, b| (a as i64 & b as i64) as f64, false),
                Opcode::BitOr(d, l, r) => do_math_untracked!(*d, *l, *r, |a, b| (a as i64 | b as i64) as f64, false),
                Opcode::Shl(d, l, r) => do_math_untracked!(*d, *l, *r, |a, b| ((a as i64) << (b as i64)) as f64, false),
                Opcode::Shr(d, l, r) => do_math_untracked!(*d, *l, *r, |a, b| ((a as i64) >> (b as i64)) as f64, false),
                Opcode::Eq(d, l, r) => {
                    let left = self.memory[self.bp + *l];
                    let right = self.memory[self.bp + *r];
                    if is_string(left) || is_string(right) {
                        let l_str = self.stringify(left);
                        let r_str = self.stringify(right);
                        self.set_reg(self.bp + *d, if l_str == r_str { 1.0 } else { 0.0 });
                    } else {
                        do_math_untracked!(*d, *l, *r, |a, b| if a == b { 1.0 } else { 0.0 }, false)
                    }
                },
                Opcode::Lt(d, l, r) => do_math_untracked!(*d, *l, *r, |a, b| if a < b { 1.0 } else { 0.0 }, false),
                Opcode::Gt(d, l, r) => do_math_untracked!(*d, *l, *r, |a, b| if a > b { 1.0 } else { 0.0 }, false),
                Opcode::And(d, l, r) => do_math_untracked!(*d, *l, *r, |a, b| if a != 0.0 && b != 0.0 { 1.0 } else { 0.0 }, false),
                Opcode::Or(d, l, r) => do_math_untracked!(*d, *l, *r, |a, b| if a != 0.0 || b != 0.0 { 1.0 } else { 0.0 }, false),
                
                Opcode::Not(d, s) => do_unary_untracked!(*d, *s, |a| if a == 0.0 { 1.0 } else { 0.0 }),
                Opcode::Neg(d, s) => do_unary_untracked!(*d, *s, |a| -a),
                Opcode::BitNot(d, s) => do_unary_untracked!(*d, *s, |a| (!(a as i64)) as f64),
                Opcode::Sin(d, s) => do_unary_untracked!(*d, *s, |a| a.sin()),
                Opcode::Cos(d, s) => do_unary_untracked!(*d, *s, |a| a.cos()),
                Opcode::Exp(d, s) => do_unary_untracked!(*d, *s, |a| a.exp()),
                Opcode::Log(d, s) => do_unary_untracked!(*d, *s, |a| a.ln()),
                Opcode::Sqrt(d, s) => do_unary_untracked!(*d, *s, |a| a.sqrt()),
                Opcode::Abs(d, s) => do_unary_untracked!(*d, *s, |a| a.abs()),

                Opcode::MatrixMul(dest, l_reg, r_reg) => {
                    let left = self.memory[self.bp + l_reg]; let right = self.memory[self.bp + r_reg];
                    self.sync_cpu(left); self.sync_cpu(right);
                    if is_array(left) && is_array(right) {
                        let l_ptr = unpack_ptr(left); let r_ptr = unpack_ptr(right);
                        let l_shape = self.tensor_shape(l_ptr); let r_shape = self.tensor_shape(r_ptr);
                        if l_shape.len() == 1 && r_shape.len() == 1 {
                            let size = l_shape[0]; let l_off = self.data_off(l_ptr); let r_off = self.data_off(r_ptr); let mut sum = 0.0; for i in 0..size { sum += self.heap[l_off + i] * self.heap[r_off + i]; } self.set_reg(self.bp + dest, sum);
                        } else {
                            let l_rows = l_shape[l_shape.len() - 2]; let l_cols = l_shape[l_shape.len() - 1]; let r_cols = r_shape[r_shape.len() - 1];
                            let mut out_shape = l_shape.clone(); let len = out_shape.len(); out_shape[len - 1] = r_cols;
                            let start = self.alloc(&out_shape); let l_off = self.data_off(l_ptr); let r_off = self.data_off(r_ptr); let d_off = self.data_off(start);
                            let out_size = l_rows * r_cols;
                            
                            if self.gpu_mode && self.gpu_ctx.has_device() {
                                let src = "extern \"C\" __global__ void zweriz_kernel(const unsigned long long* in, unsigned long long* out, const unsigned long long* shapes, size_t size) {
                                    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= size) return;
                                    const float* A = (const float*)in[0];
                                    const float* B = (const float*)in[1];
                                    float* Z = (float*)out[0];
                                    size_t l_cols = shapes[0], r_cols = shapes[1];
                                    size_t i = idx / r_cols; size_t j = idx % r_cols;
                                    float sum = 0.0f;
                                    for (size_t k = 0; k < l_cols; k++) { sum += A[i * l_cols + k] * B[k * r_cols + j]; }
                                    Z[idx] = sum;
                                }";
                                
                                let inputs = vec![
                                    GpuVal::Host { ptr: l_ptr, offset: l_off, size: l_rows * l_cols, shape_col: l_cols },
                                    GpuVal::Host { ptr: r_ptr, offset: r_off, size: l_cols * r_cols, shape_col: r_cols }
                                ];
                                let outputs = vec![(start, d_off, out_size)];
                                self.gpu_ctx.run_kernel(src, &inputs, &outputs, &self.heap, out_size);
                            } else {
                                for i in 0..l_rows { for j in 0..r_cols { let mut sum = 0.0; for k in 0..l_cols { sum += self.heap[l_off + i * l_cols + k] * self.heap[r_off + k * r_cols + j]; } self.heap[d_off + i * r_cols + j] = sum; } }
                            }
                            
                            self.set_reg(self.bp + dest, pack_array_ptr(start));
                            
                            if self.is_tracked(left) || self.is_tracked(right) {
                                self.track_tensor(start);
                                self.tape.push(TapeNode { op: TapeOp::MatMul, out_ptr: start, in_ptrs: vec![l_ptr, r_ptr] });
                                self.retain_ptr(start);
                                self.retain_ptr(l_ptr);
                                self.retain_ptr(r_ptr);
                            }
                        }
                    } else { throw!("Matrix mul requires arrays.".to_string()); }
                }

                Opcode::Print(reg) => { let val = self.memory[self.bp + reg]; let out = self.stringify(val); println!("{}", out); }
                
                Opcode::AllocArray(dest, size, start_reg) => { let mut is_nested = true; let mut sub_shape = vec![]; for i in 0..*size { let val = self.memory[self.bp + start_reg + i]; self.sync_cpu(val); if !is_array(val) { is_nested = false; break; } let ptr = unpack_ptr(val); let shp = self.tensor_shape(ptr); if i == 0 { sub_shape = shp; } else if sub_shape != shp { is_nested = false; break; } } if is_nested && *size > 0 { let mut new_shape = vec![*size]; new_shape.extend(sub_shape); let new_start = self.alloc(&new_shape); let new_offset = self.data_off(new_start); let mut ptr_offset = 0; for i in 0..*size { let val = self.memory[self.bp + start_reg + i]; let child_ptr = unpack_ptr(val); let child_size = self.tensor_data_size(child_ptr); let child_offset = self.data_off(child_ptr); for j in 0..child_size { self.heap[new_offset + ptr_offset] = self.heap[child_offset + j]; ptr_offset += 1; } } self.set_reg(self.bp + dest, pack_array_ptr(new_start)); } else { let shape = vec![*size]; let start = self.alloc(&shape); let offset = self.data_off(start); for i in 0..*size { let val = self.memory[self.bp + start_reg + i]; self.sync_cpu(val); self.heap[offset + i] = val; } self.set_reg(self.bp + dest, pack_array_ptr(start)); } }
                
                Opcode::AllocDict(dest, size, start_reg) => { 
                    if self.dict_arena.len() > 100_000 { self.run_gc(); } 
                    let mut dict = HashMap::new(); 
                    for i in 0..*size { 
                        let key_val = self.memory[self.bp + start_reg + i * 2]; 
                        let val = self.memory[self.bp + start_reg + i * 2 + 1]; 
                        let key_str = self.stringify(key_val); 
                        self.sync_cpu(val); 
                        
                        if is_array(val) { self.retain(val); } 
                        
                        dict.insert(key_str, val); 
                    } 
                    let dict_ptr = if let Some(free_idx) = self.free_dicts.pop() { 
                        self.dict_arena[free_idx] = DictObject { data: dict, marked: false, live: true }; 
                        free_idx 
                    } else { 
                        let idx = self.dict_arena.len(); 
                        self.dict_arena.push(DictObject { data: dict, marked: false, live: true }); 
                        idx 
                    }; 
                    self.set_reg(self.bp + dest, pack_dict_ptr(dict_ptr)); 
                }
                
                Opcode::Len(dest, target) => { let val = self.memory[self.bp + target]; self.sync_cpu(val); if is_array(val) { self.set_reg(self.bp + dest, self.tensor_shape(unpack_ptr(val))[0] as f64); } else if is_string(val) { self.set_reg(self.bp + dest, self.string_arena[unpack_ptr(val)].data.len() as f64); } }
                
                Opcode::LoadElement(dest, target_reg, index_reg) => { let target = self.memory[self.bp + target_reg]; let index = self.memory[self.bp + index_reg]; self.sync_cpu(target); if is_array(target) { let array_ptr = unpack_ptr(target); let size = self.tensor_data_size(array_ptr); let offset = self.data_off(array_ptr); let idx = index as usize; if idx >= size { throw!(format!("Index Error: Array index {} out of bounds", idx)); } self.set_reg(self.bp + dest, self.heap[offset + idx]); } else if is_dict(target) { let dict_ptr = unpack_ptr(target); let key_str = self.stringify(index); let val = *self.dict_arena[dict_ptr].data.get(&key_str).unwrap_or(&0.0); self.set_reg(self.bp + dest, val); } else { throw!("Type Error".to_string()); } }
                
                Opcode::StoreElement(target_reg, index_reg, val_reg) => { 
                    let target = self.memory[self.bp + target_reg]; 
                    let index = self.memory[self.bp + index_reg]; 
                    let val = self.memory[self.bp + val_reg]; 
                    self.sync_cpu(target); 
                    
                    if is_array(target) { 
                        let array_ptr = unpack_ptr(target); let size = self.tensor_data_size(array_ptr); let offset = self.data_off(array_ptr); let idx = index as usize; 
                        if idx >= size { throw!(format!("Index Error: Array index {} out of bounds", idx)); } 
                        self.heap[offset + idx] = val; 
                        self.gpu_ctx.push_elem(array_ptr, idx, val); 
                    } else if is_dict(target) { 
                        let dict_ptr = unpack_ptr(target); let key_str = self.stringify(index); 
                        
                        if let Some(&old_val) = self.dict_arena[dict_ptr].data.get(&key_str) {
                            if is_array(old_val) { self.release(old_val); }
                        }
                        if is_array(val) { self.retain(val); }
                        
                        self.dict_arena[dict_ptr].data.insert(key_str, val); 
                    } else { throw!("Type Error".to_string()); } 
                }
                
                Opcode::LoadElementND(dest, target_reg, indices_start, count) => { let target = self.memory[self.bp + target_reg]; self.sync_cpu(target); if is_array(target) { let array_ptr = unpack_ptr(target); let shape = self.tensor_shape(array_ptr); if shape.len() != *count { throw!(format!("Index Error")); } let mut flat_idx = 0; let mut stride = 1; for i in (0..*count).rev() { let idx_val = self.memory[self.bp + indices_start + i]; let idx = idx_val as usize; if idx >= shape[i] { throw!("Index out of bounds".to_string()); } flat_idx += idx * stride; stride *= shape[i]; } let offset = self.data_off(array_ptr); self.set_reg(self.bp + dest, self.heap[offset + flat_idx]); } else { throw!("Type Error".to_string()); } }
                Opcode::StoreElementND(target_reg, indices_start, count, val_reg) => { 
                    let target = self.memory[self.bp + target_reg]; let val = self.memory[self.bp + val_reg]; 
                    self.sync_cpu(target); 
                    if is_array(target) { 
                        let array_ptr = unpack_ptr(target); let shape = self.tensor_shape(array_ptr); if shape.len() != *count { throw!(format!("Index Error")); } 
                        let mut flat_idx = 0; let mut stride = 1; for i in (0..*count).rev() { let idx_val = self.memory[self.bp + indices_start + i]; let idx = idx_val as usize; if idx >= shape[i] { throw!("Index Error".to_string()); } flat_idx += idx * stride; stride *= shape[i]; } 
                        let offset = self.data_off(array_ptr); 
                        self.heap[offset + flat_idx] = val; 
                        
                        self.gpu_ctx.push_elem(array_ptr, flat_idx, val); 
                    } else { throw!("Type Error".to_string()); } 
                }
                
                Opcode::Slice(dest, target_reg, start_reg, end_reg) => { let target = self.memory[self.bp + target_reg]; let start_val = self.memory[self.bp + start_reg]; let end_val = self.memory[self.bp + end_reg]; self.sync_cpu(target); if is_array(target) { let ptr = unpack_ptr(target); let size = self.tensor_data_size(ptr); let data_offset = self.data_off(ptr); let mut start_idx = if start_val < 0.0 { 0 } else { start_val as usize }; let mut end_idx = if end_val < 0.0 { size } else { end_val as usize }; start_idx = start_idx.min(size); end_idx = end_idx.min(size); let slice_size = if end_idx > start_idx { end_idx - start_idx } else { 0 }; let start_heap = self.alloc(&[slice_size]); let dst_offset = self.data_off(start_heap); for i in start_idx..end_idx { self.heap[dst_offset + (i - start_idx)] = self.heap[data_offset + i]; } self.set_reg(self.bp + dest, pack_array_ptr(start_heap)); } else if is_string(target) { let ptr = unpack_ptr(target); let text = &self.string_arena[ptr].data; let size = text.len(); let mut start_idx = if start_val < 0.0 { 0 } else { start_val as usize }; let mut end_idx = if end_val < 0.0 { size } else { end_val as usize }; start_idx = start_idx.min(size); end_idx = end_idx.min(size); let new_str = if end_idx > start_idx { text[start_idx..end_idx].to_string() } else { "".to_string() }; let new_ptr = self.allocate_string(new_str); self.set_reg(self.bp + dest, pack_string_ptr(new_ptr)); } else { throw!("Type Error".to_string()); } }
                Opcode::Zeros(dest, rows_reg, cols_reg) | Opcode::Ones(dest, rows_reg, cols_reg) => { let rows = self.memory[self.bp + rows_reg] as usize; let cols = self.memory[self.bp + cols_reg] as usize; let shape = vec![rows, cols]; let size = rows * cols; let start = self.alloc(&shape); let offset = self.data_off(start); let fill = if let Opcode::Zeros(..) = program[pc] { 0.0 } else { 1.0 }; for i in 0..size { self.heap[offset + i] = fill; } self.set_reg(self.bp + dest, pack_array_ptr(start)); }
                
                Opcode::Blend(dest, mask_reg, true_reg, false_reg) => { 
                    let mask = self.memory[self.bp + mask_reg]; 
                    let true_val = self.memory[self.bp + true_reg]; 
                    let false_val = self.memory[self.bp + false_reg]; 
                    
                    let m_arr = is_array(mask); let t_arr = is_array(true_val); let f_arr = is_array(false_val); 
                    
                    if m_arr { 
                        let m_ptr = unpack_ptr(mask); let size = self.tensor_data_size(m_ptr); 
                        let shape = self.tensor_shape(m_ptr); let m_off = self.data_off(m_ptr); 
                        let start = self.alloc(&shape); let d_off = self.data_off(start); 
                        
                        if self.gpu_mode && self.gpu_ctx.has_device() {
                            let mut src = "extern \"C\" __global__ void zweriz_kernel(const unsigned long long* in, unsigned long long* out, const unsigned long long* shapes, size_t size) {\nsize_t idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= size) return;\nconst float* M = (const float*)in[0];\n".to_string();
                            let mut inputs = vec![GpuVal::Host { ptr: m_ptr, offset: m_off, size, shape_col: 1 }];
                            
                            if t_arr {
                                src.push_str("const float* T = (const float*)in[1];\n");
                                inputs.push(GpuVal::Host { ptr: unpack_ptr(true_val), offset: self.data_off(unpack_ptr(true_val)), size, shape_col: 1 });
                            } else { src.push_str(&format!("const float T_val = {:e}f;\n", true_val)); }
                            
                            if f_arr {
                                let f_in_idx = if t_arr { 2 } else { 1 };
                                src.push_str(&format!("const float* F = (const float*)in[{}];\n", f_in_idx));
                                inputs.push(GpuVal::Host { ptr: unpack_ptr(false_val), offset: self.data_off(unpack_ptr(false_val)), size, shape_col: 1 });
                            } else { src.push_str(&format!("const float F_val = {:e}f;\n", false_val)); }
                            
                            src.push_str("float* Z = (float*)out[0];\nfloat m = M[idx];\n");
                            src.push_str("float t = "); if t_arr { src.push_str("T[idx]"); } else { src.push_str("T_val"); }
                            src.push_str(";\nfloat f = "); if f_arr { src.push_str("F[idx]"); } else { src.push_str("F_val"); }
                            src.push_str(";\nZ[idx] = (m * t) + ((1.0f - m) * f);\n}");
                            
                            let outputs = vec![(start, d_off, size)];
                            self.gpu_ctx.run_kernel(&src, &inputs, &outputs, &self.heap, size);
                        } else {
                            self.sync_cpu(mask); self.sync_cpu(true_val); self.sync_cpu(false_val);
                            let t_off = if t_arr { self.data_off(unpack_ptr(true_val)) } else { 0 }; 
                            let f_off = if f_arr { self.data_off(unpack_ptr(false_val)) } else { 0 }; 
                            for i in 0..size { 
                                let m = self.heap[m_off + i]; 
                                let t = if t_arr { self.heap[t_off + i] } else { true_val }; 
                                let f = if f_arr { self.heap[f_off + i] } else { false_val }; 
                                self.heap[d_off + i] = (m * t) + ((1.0 - m) * f); 
                            } 
                        }
                        self.set_reg(self.bp + dest, pack_array_ptr(start)); 
                    } else if !t_arr && !f_arr { 
                        let val = if mask != 0.0 { true_val } else { false_val }; self.set_reg(self.bp + dest, val); 
                    } else { throw!("Type Error".to_string()); } 
                }

                Opcode::Reduce(dest, target_reg, axis_reg, op_type) => { 
                    let target = self.memory[self.bp + target_reg]; let axis_val = self.memory[self.bp + axis_reg]; 
                    
                    if is_array(target) { 
                        let ptr = unpack_ptr(target); let shape = self.tensor_shape(ptr); 
                        let data_offset = self.data_off(ptr); let size = self.tensor_data_size(ptr); 
                        
                        if axis_val < 0.0 { 
                            let mut res = if *op_type == 2 { f64::NEG_INFINITY } else if *op_type == 3 { f64::INFINITY } else { 0.0 }; 
                            self.sync_cpu(target);
                            for i in 0..size { let val = self.heap[data_offset + i]; if *op_type == 0 || *op_type == 1 { res += val; } else if *op_type == 2 { res = res.max(val); } else if *op_type == 3 { res = res.min(val); } } 
                            if *op_type == 1 { res /= size as f64; } 
                            self.set_reg(self.bp + dest, res); 
                        } else { 
                            let axis = axis_val as usize; if axis >= shape.len() { throw!("Reduction Error: Axis out of bounds".to_string()); } 
                            let mut out_shape = shape.clone(); out_shape.remove(axis); if out_shape.is_empty() { out_shape.push(1); } 
                            let start = self.alloc(&out_shape); let d_off = self.data_off(start); let out_size = self.tensor_data_size(start); 
                            let stride: usize = shape[axis+1..].iter().product(); let axis_dim = shape[axis]; let chunk_size = stride * axis_dim; 

                            if self.gpu_mode && self.gpu_ctx.has_device() {
                                let src = format!("extern \"C\" __global__ void zweriz_kernel(const unsigned long long* in, unsigned long long* out, const unsigned long long* shapes, size_t size) {{
                                    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x; if (out_idx >= size) return;
                                    const float* A = (const float*)in[0];
                                    float* Z = (float*)out[0];
                                    long stride = {}; long axis_dim = {}; long chunk_size = stride * axis_dim;
                                    long outer = out_idx / stride; long inner = out_idx % stride;
                                    float res = {};
                                    for (long k = 0; k < axis_dim; k++) {{
                                        long in_idx = outer * chunk_size + k * stride + inner;
                                        float val = A[in_idx];
                                        {}
                                    }}
                                    {}
                                    Z[out_idx] = res;
                                }}", stride, axis_dim, 
                                    if *op_type == 2 { "-1e38f" } else if *op_type == 3 { "1e38f" } else { "0.0f" },
                                    if *op_type == 0 || *op_type == 1 { "res += val;" } else if *op_type == 2 { "if (val > res) res = val;" } else if *op_type == 3 { "if (val < res) res = val;" } else { "" },
                                    if *op_type == 1 { "res /= (float)axis_dim;" } else { "" }
                                );
                                let inputs = vec![GpuVal::Host { ptr, offset: data_offset, size, shape_col: 1 }];
                                let outputs = vec![(start, d_off, out_size)];
                                self.gpu_ctx.run_kernel(&src, &inputs, &outputs, &self.heap, out_size);
                            } else {
                                self.sync_cpu(target);
                                for i in 0..out_size { self.heap[d_off + i] = if *op_type == 2 { f64::NEG_INFINITY } else if *op_type == 3 { f64::INFINITY } else { 0.0 }; } 
                                for i in 0..size { 
                                    let val = self.heap[data_offset + i]; let outer = i / chunk_size; let inner = i % stride; let out_idx = outer * stride + inner; 
                                    if *op_type == 0 || *op_type == 1 { self.heap[d_off + out_idx] += val; } else if *op_type == 2 { self.heap[d_off + out_idx] = self.heap[d_off + out_idx].max(val); } else if *op_type == 3 { self.heap[d_off + out_idx] = self.heap[d_off + out_idx].min(val); } 
                                } 
                                if *op_type == 1 { for i in 0..out_size { self.heap[d_off + i] /= axis_dim as f64; } } 
                            }
                            self.set_reg(self.bp + dest, pack_array_ptr(start)); 
                        } 
                    } else { throw!("Type Error".to_string()); } 
                }
                
                Opcode::DispatchGpu { cuda_src, bwd_cuda_src: _, inputs, outputs, skip_pc } => {
                    if self.gpu_ctx.has_device() {
                        let mut array_size = 1;
                        for &idx in inputs.iter().chain(outputs.iter()) { let val = self.memory[self.bp + idx]; if is_array(val) { let ptr = unpack_ptr(val); array_size = array_size.max(self.tensor_data_size(ptr)); } }
                        let mut gpu_inputs = Vec::new();
                        for &idx in inputs { let val = self.memory[self.bp + idx]; if is_array(val) { let ptr = unpack_ptr(val); let size = self.tensor_data_size(ptr); let shape = self.tensor_shape(ptr); let shape_col = *shape.last().unwrap_or(&1); gpu_inputs.push(GpuVal::Host { ptr, offset: self.data_off(ptr), size, shape_col }); } else { gpu_inputs.push(GpuVal::Scalar(val)); } }
                        let mut gpu_outputs = Vec::new();
                        for &out_idx in outputs {
                            let val = self.memory[self.bp + out_idx];
                            let prev_shape = if is_array(val) { self.tensor_shape(unpack_ptr(val)) } else { vec![array_size] };
                            let start = self.alloc(&prev_shape);
                            if is_array(val) { self.sync_cpu(val); let old_ptr = unpack_ptr(val); let old_size = self.tensor_data_size(old_ptr); let old_offset = self.data_off(old_ptr); let new_offset = self.data_off(start); let copy_len = old_size.min(self.tensor_data_size(start)); for i in 0..copy_len { self.heap[new_offset + i] = self.heap[old_offset + i]; } } else { let new_offset = self.data_off(start); let new_size = self.tensor_data_size(start); for i in 0..new_size { self.heap[new_offset + i] = val; } }
                            self.gpu_dirty.insert(start);
                            self.set_reg(self.bp + out_idx, pack_array_ptr(start));
                            let actual_size = self.tensor_data_size(start);
                            let actual_offset = self.data_off(start);
                            gpu_outputs.push((start, actual_offset, actual_size));
                        }
                        self.gpu_ctx.run_kernel(cuda_src, &gpu_inputs, &gpu_outputs, &self.heap, array_size);
                        pc = *skip_pc;
                        continue;
                    } else { if !self.fallback_warned { println!("no GPU detected, running on CPU"); self.fallback_warned = true; } }
                }
            }
            pc += 1;
        }
        Ok(())
    }
}