use rust_gpu_tools::Device;
use log::info;
use std::collections::HashMap;
use std::env;

#[derive(Copy, Clone)]
struct GPUInfo {
    max_window_size: usize,
    chunk_size_scale: f32,
    best_chunk_size_scale: f32,
    reserved_mem_ratio: f32,
}

lazy_static::lazy_static! {
    static ref GPU_INFOS: HashMap<String, GPUInfo> = {
        let mut gpu_infos : HashMap<String, GPUInfo> = vec![
            // AMD
            ("gfx1010".to_string(), GPUInfo{max_window_size: 0, chunk_size_scale: 0.0, best_chunk_size_scale: 0.0, reserved_mem_ratio: 0.2}),
            // This value was chosen to give (approximately) empirically best performance for a Radeon Pro VII.
            ("gfx906".to_string(), GPUInfo{max_window_size: 0, chunk_size_scale: 0.0, best_chunk_size_scale: 0.0, reserved_mem_ratio: 0.2}),

            // NVIDIA
            ("Quadro RTX 6000".to_string(), GPUInfo{max_window_size: 0, chunk_size_scale: 0.0, best_chunk_size_scale: 0.0, reserved_mem_ratio: 0.2}),

            ("TITAN RTX".to_string(), GPUInfo{max_window_size: 0, chunk_size_scale: 0.0, best_chunk_size_scale: 0.0, reserved_mem_ratio: 0.2}),

            ("Tesla V100".to_string(), GPUInfo{max_window_size: 0, chunk_size_scale: 0.0, best_chunk_size_scale: 0.0, reserved_mem_ratio: 0.2}),
            ("Tesla P100".to_string(), GPUInfo{max_window_size: 0, chunk_size_scale: 0.0, best_chunk_size_scale: 0.0, reserved_mem_ratio: 0.2}),
            ("Tesla T4".to_string(), GPUInfo{max_window_size: 0, chunk_size_scale: 0.0, best_chunk_size_scale: 0.0, reserved_mem_ratio: 0.2}),
            ("Quadro M5000".to_string(), GPUInfo{max_window_size: 0, chunk_size_scale: 0.0, best_chunk_size_scale: 0.0, reserved_mem_ratio: 0.2}),

            ("GeForce RTX 3090".to_string(), GPUInfo{max_window_size: 9, chunk_size_scale: 90000.0, best_chunk_size_scale: 90000.0, reserved_mem_ratio: 0.05}),
            ("GeForce RTX 3080".to_string(), GPUInfo{max_window_size: 8, chunk_size_scale: 40.0, best_chunk_size_scale: 40.0, reserved_mem_ratio: 0.0}),
            ("NVIDIA GeForce RTX 3080".to_string(), GPUInfo{max_window_size: 8, chunk_size_scale: 40.0, best_chunk_size_scale: 40.0, reserved_mem_ratio: 0.0}),
            ("GeForce RTX 3080 Ti".to_string(), GPUInfo{max_window_size: 8, chunk_size_scale: 2.0, best_chunk_size_scale: 2.0, reserved_mem_ratio: 0.2}),

            ("GeForce RTX 3070".to_string(), GPUInfo{max_window_size: 0, chunk_size_scale: 0.0, best_chunk_size_scale: 0.0, reserved_mem_ratio: 0.2}),

            ("GeForce RTX 2080 Ti".to_string(), GPUInfo{max_window_size: 9, chunk_size_scale: 29.0, best_chunk_size_scale: 29.0, reserved_mem_ratio: 0.2}),

            ("GeForce RTX 2080 SUPER".to_string(), GPUInfo{max_window_size: 0, chunk_size_scale: 0.0, best_chunk_size_scale: 0.0, reserved_mem_ratio: 0.2}),
            ("GeForce RTX 2080".to_string(), GPUInfo{max_window_size: 0, chunk_size_scale: 0.0, best_chunk_size_scale: 0.0, reserved_mem_ratio: 0.2}),
            ("GeForce RTX 2070 SUPER".to_string(), GPUInfo{max_window_size: 0, chunk_size_scale: 0.0, best_chunk_size_scale: 0.0, reserved_mem_ratio: 0.2}),

            ("GeForce GTX 1080 Ti".to_string(), GPUInfo{max_window_size: 0, chunk_size_scale: 0.0, best_chunk_size_scale: 0.0, reserved_mem_ratio: 0.2}),
            ("GeForce GTX 1080".to_string(), GPUInfo{max_window_size: 0, chunk_size_scale: 0.0, best_chunk_size_scale: 0.0, reserved_mem_ratio: 0.2}),
            ("GeForce GTX 2060".to_string(), GPUInfo{max_window_size: 0, chunk_size_scale: 0.0, best_chunk_size_scale: 0.0, reserved_mem_ratio: 0.2}),
            ("GeForce GTX 1660 Ti".to_string(), GPUInfo{max_window_size: 0, chunk_size_scale: 0.0, best_chunk_size_scale: 0.0, reserved_mem_ratio: 0.2}),
            ("GeForce GTX 1060".to_string(), GPUInfo{max_window_size: 0, chunk_size_scale: 0.0, best_chunk_size_scale: 0.0, reserved_mem_ratio: 0.2}),
            ("GeForce GTX 1650 SUPER".to_string(), GPUInfo{max_window_size: 0, chunk_size_scale: 0.0, best_chunk_size_scale: 0.0, reserved_mem_ratio: 0.2}),
            ("GeForce GTX 1650".to_string(), GPUInfo{max_window_size: 0, chunk_size_scale: 0.0, best_chunk_size_scale: 0.0, reserved_mem_ratio: 0.2}),
        ].into_iter().collect();

        if let Ok(var) = env::var("BELLMAN_CUSTOM_GPU") {
            for card in var.split(',') {
                let splitted = card.split(':').collect::<Vec<_>>();
                if splitted.len() < 2 { panic!("Invalid BELLMAN_CUSTOM_GPU!"); }
                let name = splitted[0].trim().to_string();

                let max_window_size: usize = if 3 <= splitted.len() {
                    splitted[2].trim().parse().expect("Invalid BELLMAN_CUSTOM_GPU!")
                } else {
                    10
                };
                let chunk_size_scale: f32 = if 4 <= splitted.len() {
                    splitted[3].trim().parse().expect("Invalid BELLMAN_CUSTOM_GPU!")
                } else {
                    2.0
                };
                let best_chunk_size_scale: f32 = if 5 <= splitted.len() {
                    splitted[4].trim().parse().expect("Invalid BELLMAN_CUSTOM_GPU!")
                } else {
                    2.0
                };
                let reserved_mem_ratio : f32 = if 6 <= splitted.len() {
                    splitted[5].trim().parse().expect("Invalid BELLMAN_CUSTOM_GPU!")
                } else {
                    0.2
                };
                info!("Adding \"{}\" to GPU list: max window size {} chunk size scale {} best chunk size scale {} reserved mem ratio {}",
                      name, max_window_size, chunk_size_scale, best_chunk_size_scale, reserved_mem_ratio);
                gpu_infos.
                    insert(
                        name,
                        GPUInfo{
                            max_window_size,
                            chunk_size_scale,
                            best_chunk_size_scale,
                            reserved_mem_ratio,
                        });
            }
        }

        gpu_infos
    };
}

pub fn get_max_window_size(d: &Device) -> usize {
    let name = d.name();
    match GPU_INFOS.get(&name[..]) {
        Some(&info) => info.max_window_size,
        None => 10,
    }
}

pub fn get_chunk_size_scale(d: &Device) -> f32 {
    let name = d.name();
    match GPU_INFOS.get(&name[..]) {
        Some(&info) => info.chunk_size_scale,
        None => 2.0,
    }
}

pub fn get_best_chunk_size_scale(d: &Device) -> f32 {
    let name = d.name();
    match GPU_INFOS.get(&name[..]) {
        Some(&info) => info.best_chunk_size_scale,
        None => 2.0,
    }
}

pub fn get_reserved_mem_ratio(d: &Device) -> f32 {
    let name = d.name();
    match GPU_INFOS.get(&name[..]) {
        Some(&info) => info.reserved_mem_ratio,
        None => 0.2,
    }
}

pub fn dump_device_list() {
    for d in Device::all() {
        info!("Device: {:?}", d);
    }
}

#[cfg(any(feature = "cuda", feature = "opencl"))]
#[test]
pub fn test_list_devices() {
    let _ = env_logger::try_init();
    dump_device_list();
}
