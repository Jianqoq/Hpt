use std::collections::HashMap;
use std::io::Read;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use regex::Regex;
use walkdir::WalkDir;

fn main() {
    if !cfg!(feature = "cuda") {
        return;
    }
    let caps = compute_cap();
    let cu_files = find_cu_files(Path::new("src")).expect("find cu files");
    let h_files = find_h_files(Path::new("src")).expect("find cuh files");
    for cu_file in &cu_files {
        println!("cargo:rerun-if-changed={}", cu_file.display());
    }
    for h_file in &h_files {
        println!("cargo:rerun-if-changed={}", h_file.display());
    }
    // create OUT_DIR if not exists
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let generated_constants = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"))
        .join("generated_constants.rs");
    let out_dir = Path::new(&out_dir);
    if !out_dir.exists() {
        std::fs::create_dir_all(out_dir).expect("create OUT_DIR");
    }
    let mut functions_map = HashMap::new();
    for cap in &caps {
        functions_map.insert(cap, HashMap::<String, (usize,)>::new());
    }

    let mut buffer = String::new();

    for cu_file in cu_files {
        let obj_files = compile_cu(&cu_file, out_dir, &caps).expect("compile failed");
        let mut cap_map = phf_codegen::Map::new();
        for (idx, name) in obj_files.into_iter().enumerate() {
            let name_upper_case = name.to_uppercase();
            cap_map.entry(
                caps[idx],
                &format!(
                    "({}, &{}, &{})",
                    &name_upper_case,
                    format!("{}_REG_INFO", &name_upper_case),
                    format!("{}_FUNC_LIST", &name_upper_case)
                ),
            );
            let content = std::fs::read_to_string(out_dir.join(format!("{}.ptx", name)))
                .expect("read ptx file");
            let mut map = phf_codegen::Map::new();
            let reg_info = count_registers(&content);
            let mut func_list = Vec::new();
            for (name, (pred, b16, b32, b64)) in reg_info {
                func_list.push(format!("\"{}\"", name));
                map.entry(
                    name,
                    &format!(
                        "RegisterInfo {{ pred: {}, b16: {}, b32: {}, b64: {} }}",
                        pred, b16, b32, b64
                    ),
                );
            }
            buffer.push_str(&format!(
                "pub const {}_REG_INFO: phf::Map<&'static str, RegisterInfo> = {};\n",
                &name_upper_case,
                map.build()
            ));
            buffer.push_str(&format!(
                "pub const {}: &str = include_str!(concat!(env!(\"OUT_DIR\"), \"/{}.ptx\"));\n",
                &name_upper_case, name
            ));
            buffer.push_str(&format!(
                "pub const {}_FUNC_LIST: [&str; {}] = [{}];\n",
                &name_upper_case,
                func_list.len(),
                func_list.join(", ")
            ));
        }
        let file_stem = cu_file
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| format!("Invalid file stem for {:?}", cu_file))
            .expect(format!("Invalid file stem for {:?}", cu_file).as_str());
        buffer.push_str(&format!(
            "pub const {}: phf::Map<usize, (&'static str, &'static phf::Map<&'static str, RegisterInfo>, &'static [&str])> = {};",
            file_stem.to_uppercase(),
            cap_map.build()
        ));
    }
    if let Ok(mut file) = std::fs::File::open(&generated_constants) {
        let mut content = String::new();
        file.read_to_string(&mut content).unwrap();
        let mut file =
            std::fs::File::create(&generated_constants).expect("create generated.rs");
        file.write_all(buffer.as_bytes()).unwrap();
    } else {
        let mut file =
            std::fs::File::create(&generated_constants).expect("create generated.rs");
        file.write_all(buffer.as_bytes()).unwrap();
    }
}

pub(crate) fn count_registers(ptx: &str) -> HashMap<String, (usize, usize, usize, usize)> {
    let mut reg_counts = HashMap::new();

    // 匹配函数名和寄存器声明
    let func_re = Regex::new(r"\.visible \.entry (\w+)\(").unwrap();
    let pred_re = Regex::new(r"\.reg \.pred\s+%p<(\d+)>").unwrap();
    let b16_re = Regex::new(r"\.reg \.b16\s+%rs<(\d+)>").unwrap();
    let b32_re = Regex::new(r"\.reg \.b32\s+%r<(\d+)>").unwrap();
    let b64_re = Regex::new(r"\.reg \.b64\s+%rd<(\d+)>").unwrap();
    let mut current_func = String::new();

    for line in ptx.lines() {
        if let Some(cap) = func_re.captures(line) {
            current_func = cap[1].to_string();
        } else if !current_func.is_empty() {
            if let Some(cap) = pred_re.captures(line) {
                let pred_count = cap[1].parse::<usize>().unwrap();
                reg_counts
                    .entry(current_func.clone())
                    .or_insert((0, 0, 0, 0))
                    .0 = pred_count;
            }
            if let Some(cap) = b16_re.captures(line) {
                let b16_count = cap[1].parse::<usize>().unwrap();
                reg_counts
                    .entry(current_func.clone())
                    .or_insert((0, 0, 0, 0))
                    .1 = b16_count;
            }
            if let Some(cap) = b32_re.captures(line) {
                let b32_count = cap[1].parse::<usize>().unwrap();
                reg_counts
                    .entry(current_func.clone())
                    .or_insert((0, 0, 0, 0))
                    .2 = b32_count;
            }
            if let Some(cap) = b64_re.captures(line) {
                let b64_count = cap[1].parse::<usize>().unwrap();
                reg_counts
                    .entry(current_func.clone())
                    .or_insert((0, 0, 0, 0))
                    .3 = b64_count;
            }
        }
    }

    reg_counts
}

fn find_cu_files(dir: &Path) -> Result<Vec<PathBuf>, std::io::Error> {
    let mut cu_files = Vec::new();
    for entry in WalkDir::new(dir).follow_links(true) {
        let entry = entry?;
        let path = entry.path().to_owned();
        if path.extension().and_then(|s| s.to_str()) == Some("cu") {
            cu_files.push(path);
        }
    }
    Ok(cu_files)
}

fn find_h_files(dir: &Path) -> Result<Vec<PathBuf>, std::io::Error> {
    let mut h_files = Vec::new();
    for entry in WalkDir::new(dir).follow_links(true) {
        let entry = entry?;
        let path = entry.path().to_owned();
        if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
            if ext == "cuh" || ext == "h" {
                h_files.push(path);
            }
        }
    }
    Ok(h_files)
}

fn compile_cu(cu_file: &Path, out_dir: &Path, caps: &[u32]) -> Result<Vec<String>, String> {
    let mut obj_files = Vec::new();
    for cap in caps {
        let file_stem = cu_file
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| format!("Invalid file stem for {:?}", cu_file))?;

        let name = format!("{}_{}", file_stem, cap);
        let obj_file = out_dir.join(format!("{}.ptx", name));
        let obj_file_exists = obj_file.exists();
        if obj_file_exists {
            let obj_time = obj_file
                .metadata()
                .and_then(|m| m.modified())
                .expect("Failed to get obj file time");
            let src_newer = cu_file
                .metadata()
                .and_then(|m| m.modified())
                .map(|t| t > obj_time)
                .unwrap_or(true);
            let header_files = get_dependencies(cu_file)?;
            let headers_newer = header_files.iter().any(|h| {
                h.metadata()
                    .and_then(|m| m.modified())
                    .map(|t| t > obj_time)
                    .unwrap_or(true)
            });
            if !src_newer && !headers_newer {
                obj_files.push(name);
                continue;
            }
        }
        let mut cmd = Command::new("nvcc");
        cmd.arg("-ptx")
            .arg("-O3")
            .arg("-allow-unsupported-compiler")
            .arg("--extended-lambda")
            .arg(cu_file.to_str().unwrap())
            .arg("-o")
            .arg(obj_file.to_str().unwrap())
            .arg(format!("-arch=sm_{}", cap));

        let status = cmd
            .status()
            .map_err(|e| format!("Failed to execute nvcc: {}", e))?;

        if !status.success() {
            return Err(format!(
                "nvcc failed to compile {:?} with status: {}",
                cu_file, status
            ));
        }
        obj_files.push(name);
    }

    Ok(obj_files)
}

fn get_dependencies(cu_file: &Path) -> Result<Vec<PathBuf>, String> {
    let mut cmd = Command::new("nvcc");
    cmd.arg("-M").arg(cu_file.to_str().unwrap());

    let output = cmd
        .output()
        .map_err(|e| format!("Failed to get dependencies: {}", e))?;

    let deps = String::from_utf8_lossy(&output.stdout);

    let src_dir = std::env::current_dir()
        .expect("get current dir failed")
        .join("src")
        .canonicalize()
        .map_err(|e| e.to_string())?;

    let deps: Vec<PathBuf> = deps
        .split_whitespace()
        .filter(|path| {
            let ext = Path::new(path).extension().and_then(|s| s.to_str());
            ext == Some("h") || ext == Some("cuh") || ext == Some("hpp")
        })
        .map(|path| PathBuf::from(path.replace("\\", "")))
        .filter(|path| {
            if let Ok(abs_path) = path.canonicalize() {
                abs_path.starts_with(&src_dir)
            } else {
                false
            }
        })
        .collect();

    Ok(deps)
}

fn compute_cap() -> Vec<u32> {
    let out = std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=compute_cap")
        .arg("--format=csv")
        .output()
        .expect("cannot find `nvidia-smi` in PATH.");
    let out = std::str::from_utf8(&out.stdout).expect("stdout is not a utf8 string");
    let mut lines = out.lines();
    assert_eq!(lines.next().expect("missing line in stdout"), "compute_cap");
    let caps = lines
        .into_iter()
        .map(|s| s.replace('.', "").parse::<u32>().unwrap())
        .collect::<Vec<u32>>();
    caps
}
