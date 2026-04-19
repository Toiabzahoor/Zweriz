
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(not(feature = "cuda"))]
pub use mock::*;

#[derive(Clone, Copy)]
pub enum GpuVal {
    Host { ptr: usize, offset: usize, size: usize, shape_col: usize },
    Scalar(f64),
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::collections::HashMap;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::sync::Arc;
    use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig, CudaSlice, DevicePtr, DeviceSlice};
    use cudarc::nvrtc::compile_ptx;
    use super::GpuVal;

    pub struct Gpu {
        device: Option<Arc<CudaDevice>>,
        kernel_cache: HashMap<u64, String>,

        gpu_mem: HashMap<usize, CudaSlice<f32>>,
    }

    impl Gpu {
        pub fn new() -> Self {
            match CudaDevice::new(0) {
                Ok(dev) => {
                    println!("GPU ready.");
                    Gpu { device: Some(dev), kernel_cache: HashMap::new(), gpu_mem: HashMap::new() }
                },
                Err(_) => {
                    Gpu { device: None, kernel_cache: HashMap::new(), gpu_mem: HashMap::new() }
                }
            }
        }

        pub fn has_device(&self) -> bool { self.device.is_some() }

        pub fn pull(&mut self, ptr: usize, host_slice: &mut [f64]) {
            if let Some(dev) = &self.device {
                if let Some(d_data) = self.gpu_mem.get(&ptr) {

                    let copy_len = std::cmp::min(d_data.len(), host_slice.len());
                    if copy_len == 0 { return; }

                    let mut temp = vec![0.0f32; copy_len];

                    let sub_slice = d_data.slice(0..copy_len);
                    dev.dtoh_sync_copy_into(&sub_slice, &mut temp).unwrap();

                    for (i, val) in temp.iter().enumerate() {
                        host_slice[i] = *val as f64;
                    }
                }
            }
        }

        pub fn invalidate(&mut self, ptr: usize) { self.gpu_mem.remove(&ptr); }

        pub fn push_elem(&mut self, ptr: usize, index: usize, value: f64) {
            if let Some(dev) = &self.device {
                if let Some(d_data) = self.gpu_mem.get_mut(&ptr) {
                    if let Some(mut sub_slice) = d_data.try_slice_mut(index..index + 1) {
                        let _ = dev.htod_sync_copy_into(&[value as f32], &mut sub_slice);
                    }
                }
            }
        }

        pub fn run_kernel(
            &mut self,
            cuda_src: &str,
            inputs: &[GpuVal],
            outputs: &[(usize, usize, usize)],
            heap: &[f64],
            max_size: usize,
        ) {
            let dev = self.device.as_ref().expect("CUDA device not initialized");

            let mut hasher = DefaultHasher::new();
            cuda_src.hash(&mut hasher);
            let hash = hasher.finish();

            let module_name = format!("zweriz_kernel_{}", hash);

            if !self.kernel_cache.contains_key(&hash) {
                let ptx = compile_ptx(cuda_src).expect("CUDA C++ Compilation Error");
                dev.load_ptx(ptx, &module_name, &["zweriz_kernel"]).expect("Failed to load PTX module");
                self.kernel_cache.insert(hash, module_name.clone());
            }

            let f = dev.get_func(&module_name, "zweriz_kernel").expect("Failed to find kernel function");

            let mut temp_slices = Vec::new();
            let mut in_ptrs = Vec::with_capacity(inputs.len());
            let mut shape_cols = Vec::with_capacity(inputs.len());

            for inp in inputs {
                match inp {
                    GpuVal::Host { ptr, offset, size, shape_col } => {
                        if !self.gpu_mem.contains_key(ptr) {

                            let f32_vec: Vec<f32> = heap[*offset .. *offset + *size].iter().map(|&v| v as f32).collect();
                            let d_data = dev.htod_copy(f32_vec).unwrap();
                            self.gpu_mem.insert(*ptr, d_data);
                        }
                        in_ptrs.push(*self.gpu_mem.get(ptr).unwrap().device_ptr() as u64);
                        shape_cols.push(*shape_col as u64);
                    },
                    GpuVal::Scalar(val) => {
                        let d_data = dev.htod_copy(vec![*val as f32]).unwrap();
                        in_ptrs.push(*d_data.device_ptr() as u64);
                        shape_cols.push(1);
                        temp_slices.push(d_data);
                    }
                }
            }

            let mut out_ptrs = Vec::with_capacity(outputs.len());
            for &(out_ptr, _out_offset, out_size) in outputs {
                if !self.gpu_mem.contains_key(&out_ptr) || self.gpu_mem.get(&out_ptr).unwrap().len() != out_size {
                    let d_data: CudaSlice<f32> = dev.alloc_zeros::<f32>(out_size).unwrap();
                    self.gpu_mem.insert(out_ptr, d_data);
                }
                out_ptrs.push(*self.gpu_mem.get(&out_ptr).unwrap().device_ptr() as u64);
            }

            let d_in_ptrs = dev.htod_copy(in_ptrs).unwrap();
            let mut d_out_ptrs = dev.htod_copy(out_ptrs).unwrap();
            let d_shapes = dev.htod_copy(shape_cols).unwrap();

            let cfg = LaunchConfig { grid_dim: ((max_size as u32 + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            unsafe { f.launch(cfg, (&d_in_ptrs, &mut d_out_ptrs, &d_shapes, max_size)).unwrap(); }
        }

        pub fn run_named_kernel(
            &mut self,
            cuda_src: &str,
            kernel_name: &'static str,
            inputs: &[GpuVal],
            outputs: &[(usize, usize, usize)],
            heap: &[f64],
            max_size: usize,
        ) {
            let dev = self.device.as_ref().expect("CUDA device not initialized");

            let mut hasher = DefaultHasher::new();
            cuda_src.hash(&mut hasher);
            kernel_name.hash(&mut hasher);
            let hash = hasher.finish();

            let module_name = format!("zweriz_module_{}", hash);

            if !self.kernel_cache.contains_key(&hash) {
                let ptx = compile_ptx(cuda_src).expect("CUDA C++ Compilation Error");
                dev.load_ptx(ptx, &module_name, &[kernel_name]).expect("Failed to load PTX module");
                self.kernel_cache.insert(hash, module_name.clone());
            }

            let f = dev.get_func(&module_name, kernel_name).expect("Failed to find kernel function");

            let mut temp_slices = Vec::new();
            let mut in_ptrs = Vec::with_capacity(inputs.len());
            let mut shape_cols = Vec::with_capacity(inputs.len());

            for inp in inputs {
                match inp {
                    GpuVal::Host { ptr, offset, size, shape_col } => {
                        if !self.gpu_mem.contains_key(ptr) {
                            let f32_vec: Vec<f32> = heap[*offset .. *offset + *size].iter().map(|&v| v as f32).collect();
                            let d_data = dev.htod_copy(f32_vec).unwrap();
                            self.gpu_mem.insert(*ptr, d_data);
                        }
                        in_ptrs.push(*self.gpu_mem.get(ptr).unwrap().device_ptr() as u64);
                        shape_cols.push(*shape_col as u64);
                    },
                    GpuVal::Scalar(val) => {
                        let d_data = dev.htod_copy(vec![*val as f32]).unwrap();
                        in_ptrs.push(*d_data.device_ptr() as u64);
                        shape_cols.push(1);
                        temp_slices.push(d_data);
                    }
                }
            }

            let mut out_ptrs = Vec::with_capacity(outputs.len());
            for &(out_ptr, _out_offset, out_size) in outputs {
                if !self.gpu_mem.contains_key(&out_ptr) || self.gpu_mem.get(&out_ptr).unwrap().len() != out_size {
                    let d_data: cudarc::driver::CudaSlice<f32> = dev.alloc_zeros::<f32>(out_size).unwrap();
                    self.gpu_mem.insert(out_ptr, d_data);
                }
                out_ptrs.push(*self.gpu_mem.get(&out_ptr).unwrap().device_ptr() as u64);
            }

            let d_in_ptrs = dev.htod_copy(in_ptrs).unwrap();
            let mut d_out_ptrs = dev.htod_copy(out_ptrs).unwrap();
            let d_shapes = dev.htod_copy(shape_cols).unwrap();

            let cfg = cudarc::driver::LaunchConfig { grid_dim: ((max_size as u32 + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            unsafe { f.launch(cfg, (&d_in_ptrs, &mut d_out_ptrs, &d_shapes, max_size)).unwrap(); }
        }
    }
}

#[cfg(not(feature = "cuda"))]
mod mock {
    use super::GpuVal;
    pub struct Gpu {}
    impl Gpu {
        pub fn new() -> Self { Gpu {} }
        pub fn has_device(&self) -> bool { false }
        pub fn pull(&mut self, _ptr: usize, _host_slice: &mut [f64]) {}
        pub fn invalidate(&mut self, _ptr: usize) {}
        pub fn push_elem(&mut self, _ptr: usize, _index: usize, _value: f64) {}
        pub fn run_kernel(&mut self, _src: &str, _in: &[GpuVal], _out: &[(usize, usize, usize)], _heap: &[f64], _size: usize) {}
        pub fn run_named_kernel(&mut self, _src: &str, _name: &'static str, _in: &[GpuVal], _out: &[(usize, usize, usize)], _heap: &[f64], _size: usize) {}
    }
}