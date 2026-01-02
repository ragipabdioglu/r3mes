// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod process_manager;
mod hardware_check;
mod keychain;
mod updater;
mod model_downloader;
mod setup_checker;
mod status_monitor;
mod log_reader;
mod wallet;
mod installer;
mod engine_downloader;
mod config;
mod websocket_client;
mod debug;

use commands::*;
use tauri::{Manager, SystemTray, SystemTrayEvent, SystemTrayMenu, WindowEvent};
use log::{info, error};

fn main() {
    let tray_menu = SystemTrayMenu::new();
    let system_tray = SystemTray::new().with_menu(tray_menu);

    tauri::Builder::default()
        .system_tray(system_tray)
        .on_system_tray_event(|app, event| {
            if let SystemTrayEvent::LeftClick {
                position: _,
                size: _,
                ..
            } = event
            {
                if let Some(window) = app.get_window("main") {
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            }
        })
        .on_window_event(|event| {
            if let WindowEvent::CloseRequested { api, .. } = event.event() {
                // Prevent immediate close to allow cleanup
                api.prevent_close();
                
                let window = event.window().clone();
                
                // Cleanup all child processes before closing
                use crate::commands::cleanup_all_processes;
                tauri::async_runtime::spawn(async move {
                    info!("Cleaning up processes before exit...");
                    if let Err(e) = cleanup_all_processes().await {
                        error!("Failed to cleanup processes: {}", e);
                    }
                    
                    // Disconnect WebSocket
                    let ws_client = websocket_client::get_ws_client();
                    if let Some(ref mut client) = *ws_client.write().await {
                        client.disconnect().await;
                    }
                    
                    info!("Cleanup complete, closing window");
                    let _ = window.close();
                });
            }
        })
        .invoke_handler(tauri::generate_handler![
            start_node,
            stop_node,
            start_miner,
            stop_miner,
            start_ipfs,
            stop_ipfs,
            start_serving,
            stop_serving,
            start_validator,
            stop_validator,
            start_proposer,
            stop_proposer,
            get_status,
            get_logs,
            get_logs_tail,
            get_logs_by_level,
            export_logs,
            check_hardware,
            is_first_run,
            mark_setup_complete,
            get_wallet_info,
            create_wallet,
            import_wallet_from_private_key,
            import_wallet_from_mnemonic,
            export_wallet,
            get_chain_status,
            get_ipfs_status,
            get_model_status,
            get_mining_stats,
            get_transaction_history,
            open_dashboard,
            check_firewall_ports,
            ensure_engine_ready,
            download_engine,
            register_node_roles,
            get_config,
            save_config,
            reset_config_to_defaults,
            check_setup_status,
            get_setup_steps,
            validate_component,
            get_setup_progress,
            collect_debug_info,
            export_debug_info,
            get_troubleshooting_recommendations
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

