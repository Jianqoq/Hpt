pub(crate) fn create_file(path: std::path::PathBuf, ext: &str) -> std::io::Result<std::fs::File> {
    if let Some(extension) = path.extension() {
        if extension == ext {
            std::fs::File::create(path)
        } else {
            std::fs::File::create(format!("{}.{ext}", path.to_str().unwrap()))
        }
    } else {
        std::fs::File::create(format!("{}.{ext}", path.to_str().unwrap()))
    }
}
