// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

// Learn more about Tauri commands at https://tauri.app/v1/guides/features/command
#[tauri::command]
fn get_graph_data() -> &'static serde_json::Value {
    todo!("implement get_graph_data")
}

#[tauri::command]
fn get_hir_data() -> &'static serde_json::Value {
    todo!("implement get_graph_data")
}

#[tauri::command]
fn get_mir_data() -> &'static serde_json::Value {
    todo!("implement get_graph_data")
}

fn main() {
    tauri::Builder
        ::default()
        .invoke_handler(tauri::generate_handler![get_graph_data, get_hir_data, get_mir_data])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
