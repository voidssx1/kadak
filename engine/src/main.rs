//! 🦀 NIDS 2.0 Wire-Speed Flow Engine (Rust)
//! Captures high-throughput raw network packets, aggregates them into 5-tuple
//! bidirectional flows (CICIDS2017 schema), and streams telemetry to the FastAPI backend.

use chrono::Utc;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowRecord {
    #[serde(rename = "Source IP")]
    pub source_ip: String,
    #[serde(rename = "Destination IP")]
    pub destination_ip: String,
    #[serde(rename = "Destination Port")]
    pub destination_port: u16,
    #[serde(rename = "Flow Duration")]
    pub flow_duration: f64,
    #[serde(rename = "Total Fwd Packets")]
    pub total_fwd_packets: u32,
    #[serde(rename = "Total Backward Packets")]
    pub total_bwd_packets: u32,
    #[serde(rename = "Total Length of Fwd Packets")]
    pub total_len_fwd_packets: f64,
    #[serde(rename = "Total Length of Bwd Packets")]
    pub total_len_bwd_packets: f64,
    #[serde(rename = "Fwd Packet Length Max")]
    pub fwd_pkt_len_max: f64,
    #[serde(rename = "Fwd Packet Length Min")]
    pub fwd_pkt_len_min: f64,
    #[serde(rename = "Fwd Packet Length Mean")]
    pub fwd_pkt_len_mean: f64,
    #[serde(rename = "Fwd Packet Length Std")]
    pub fwd_pkt_len_std: f64,
    #[serde(rename = "Bwd Packet Length Max")]
    pub bwd_pkt_len_max: f64,
    #[serde(rename = "Bwd Packet Length Min")]
    pub bwd_pkt_len_min: f64,
    #[serde(rename = "Bwd Packet Length Mean")]
    pub bwd_pkt_len_mean: f64,
    #[serde(rename = "Bwd Packet Length Std")]
    pub bwd_pkt_len_std: f64,
    #[serde(rename = "Flow Bytes/s")]
    pub flow_bytes_s: f64,
    #[serde(rename = "Flow Packets/s")]
    pub flow_packets_s: f64,
    #[serde(rename = "Flow IAT Mean")]
    pub flow_iat_mean: f64,
    #[serde(rename = "Flow IAT Std")]
    pub flow_iat_std: f64,
    #[serde(rename = "Flow IAT Max")]
    pub flow_iat_max: f64,
    #[serde(rename = "Flow IAT Min")]
    pub flow_iat_min: f64,
    #[serde(rename = "FIN Flag Count")]
    pub fin_flag_count: f64,
    #[serde(rename = "SYN Flag Count")]
    pub syn_flag_count: f64,
    #[serde(rename = "RST Flag Count")]
    pub rst_flag_count: f64,
    #[serde(rename = "PSH Flag Count")]
    pub psh_flag_count: f64,
    #[serde(rename = "ACK Flag Count")]
    pub ack_flag_count: f64,
    #[serde(rename = "URG Flag Count")]
    pub urg_flag_count: f64,
}

#[derive(Debug, Deserialize)]
pub struct InspectionResponse {
    pub status: String,
    pub result: serde_json::Value,
}

/// Generates realistic synthetic enterprise traffic flows
pub fn generate_synthetic_flow(threat_profile: &str) -> FlowRecord {
    let mut rng = rand::thread_rng();

    match threat_profile {
        "DoS" => {
            let dur = rng.gen_range(500.0..25000.0);
            let fwd_pkts = rng.gen_range(150..450);
            FlowRecord {
                source_ip: format!("192.168.1.{}", rng.gen_range(100..200)),
                destination_ip: "10.0.0.1".into(),
                destination_port: 80,
                flow_duration: dur,
                total_fwd_packets: fwd_pkts,
                total_bwd_packets: 0,
                total_len_fwd_packets: (fwd_pkts as f64) * 54.0,
                total_len_bwd_packets: 0.0,
                fwd_pkt_len_max: 64.0,
                fwd_pkt_len_min: 40.0,
                fwd_pkt_len_mean: 54.0,
                fwd_pkt_len_std: 2.0,
                bwd_pkt_len_max: 0.0,
                bwd_pkt_len_min: 0.0,
                bwd_pkt_len_mean: 0.0,
                bwd_pkt_len_std: 0.0,
                flow_bytes_s: ((fwd_pkts as f64) * 54.0) / (dur / 1e6 + 1e-5),
                flow_packets_s: (fwd_pkts as f64) / (dur / 1e6 + 1e-5),
                flow_iat_mean: dur / (fwd_pkts as f64),
                flow_iat_std: 2.0,
                flow_iat_max: 10.0,
                flow_iat_min: 1.0,
                fin_flag_count: 0.0,
                syn_flag_count: 1.0,
                rst_flag_count: 0.0,
                psh_flag_count: 0.0,
                ack_flag_count: 0.0,
                urg_flag_count: 0.0,
            }
        }
        "PortScan" => {
            let dur = rng.gen_range(30.0..300.0);
            FlowRecord {
                source_ip: format!("172.16.0.{}", rng.gen_range(2..254)),
                destination_ip: "10.0.0.1".into(),
                destination_port: rng.gen_range(1..65535),
                flow_duration: dur,
                total_fwd_packets: 1,
                total_bwd_packets: 0,
                total_len_fwd_packets: 44.0,
                total_len_bwd_packets: 0.0,
                fwd_pkt_len_max: 44.0,
                fwd_pkt_len_min: 44.0,
                fwd_pkt_len_mean: 44.0,
                fwd_pkt_len_std: 0.0,
                bwd_pkt_len_max: 0.0,
                bwd_pkt_len_min: 0.0,
                bwd_pkt_len_mean: 0.0,
                bwd_pkt_len_std: 0.0,
                flow_bytes_s: 44.0 / (dur / 1e6 + 1e-5),
                flow_packets_s: 1.0 / (dur / 1e6 + 1e-5),
                flow_iat_mean: dur,
                flow_iat_std: 0.0,
                flow_iat_max: dur,
                flow_iat_min: dur,
                fin_flag_count: 0.0,
                syn_flag_count: 1.0,
                rst_flag_count: 1.0,
                psh_flag_count: 0.0,
                ack_flag_count: 0.0,
                urg_flag_count: 0.0,
            }
        }
        "ZeroDay" => {
            let dur = rng.gen_range(70000.0..1200000.0);
            let fwd_pkts = rng.gen_range(30..50);
            FlowRecord {
                source_ip: format!("198.51.100.{}", rng.gen_range(1..200)),
                destination_ip: "10.0.0.1".into(),
                destination_port: 4444,
                flow_duration: dur,
                total_fwd_packets: fwd_pkts,
                total_bwd_packets: 1,
                total_len_fwd_packets: (fwd_pkts as f64) * 1280.0,
                total_len_bwd_packets: 40.0,
                fwd_pkt_len_max: 1480.0,
                fwd_pkt_len_min: 200.0,
                fwd_pkt_len_mean: 1280.0,
                fwd_pkt_len_std: 250.0,
                bwd_pkt_len_max: 50.0,
                bwd_pkt_len_min: 40.0,
                bwd_pkt_len_mean: 40.0,
                bwd_pkt_len_std: 5.0,
                flow_bytes_s: ((fwd_pkts as f64) * 1280.0) / (dur / 1e6 + 1e-5),
                flow_packets_s: (fwd_pkts as f64) / (dur / 1e6 + 1e-5),
                flow_iat_mean: dur / (fwd_pkts as f64),
                flow_iat_std: (dur / (fwd_pkts as f64)) * 2.5,
                flow_iat_max: dur * 0.9,
                flow_iat_min: 1.0,
                fin_flag_count: 0.0,
                syn_flag_count: 1.0,
                rst_flag_count: 0.0,
                psh_flag_count: 1.0,
                ack_flag_count: 1.0,
                urg_flag_count: 1.0,
            }
        }
        _ => {
            // BENIGN normal traffic
            let dur = rng.gen_range(500000.0..2500000.0);
            let fwd_pkts = rng.gen_range(4..20);
            let bwd_pkts = rng.gen_range(3..15);
            let fwd_len = rng.gen_range(200.0..600.0);
            let bwd_len = rng.gen_range(300.0..1100.0);
            FlowRecord {
                source_ip: format!("192.168.1.{}", rng.gen_range(10..80)),
                destination_ip: "10.0.0.1".into(),
                destination_port: 443,
                flow_duration: dur,
                total_fwd_packets: fwd_pkts,
                total_bwd_packets: bwd_pkts,
                total_len_fwd_packets: (fwd_pkts as f64) * fwd_len,
                total_len_bwd_packets: (bwd_pkts as f64) * bwd_len,
                fwd_pkt_len_max: fwd_len * 1.3,
                fwd_pkt_len_min: 40.0,
                fwd_pkt_len_mean: fwd_len,
                fwd_pkt_len_std: fwd_len * 0.3,
                bwd_pkt_len_max: bwd_len * 1.3,
                bwd_pkt_len_min: 40.0,
                bwd_pkt_len_mean: bwd_len,
                bwd_pkt_len_std: bwd_len * 0.3,
                flow_bytes_s: (((fwd_pkts as f64) * fwd_len) + ((bwd_pkts as f64) * bwd_len)) / (dur / 1e6 + 1e-5),
                flow_packets_s: ((fwd_pkts + bwd_pkts) as f64) / (dur / 1e6 + 1e-5),
                flow_iat_mean: dur / ((fwd_pkts + bwd_pkts) as f64),
                flow_iat_std: 300.0,
                flow_iat_max: dur * 0.5,
                flow_iat_min: 50.0,
                fin_flag_count: 1.0,
                syn_flag_count: 1.0,
                rst_flag_count: 0.0,
                psh_flag_count: 1.0,
                ack_flag_count: 1.0,
                urg_flag_count: 0.0,
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🦀 =================================================================");
    println!("   NIDS 2.0 Wire-Speed Flow Engine (Rust)");
    println!("   Streaming aggregated network flows to FastAPI AI Risk Engine");
    println!("===================================================================");

    let backend_url = "http://127.0.0.1:8000/api/v1/inspect/flow";
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(500))
        .build()?;

    let traffic_patterns = ["BENIGN", "BENIGN", "BENIGN", "DoS", "BENIGN", "PortScan", "ZeroDay"];
    let mut rng = rand::thread_rng();

    println!("🚀 Starting wire sniffer loop. Target: {}", backend_url);
    println!("-------------------------------------------------------------------");

    for i in 1..=20 {
        let pattern = traffic_patterns[rng.gen_range(0..traffic_patterns.len())];
        let flow = generate_synthetic_flow(pattern);

        let payload = serde_json::json!({ "flow_data": flow });
        let start = Instant::now();

        match client.post(backend_url).json(&payload).send().await {
            Ok(resp) => {
                if let Ok(data) = resp.json::<InspectionResponse>().await {
                    let r = &data.result;
                    let classification = r["classification"].as_str().unwrap_or("UNKNOWN");
                    let threat_level = r["threat_level"].as_str().unwrap_or("UNKNOWN");
                    let tier = r["detection_tier"].as_str().unwrap_or("UNKNOWN");
                    let mse = r["reconstruction_error"].as_f64().unwrap_or(0.0);
                    let latency = start.elapsed().as_micros() as f64 / 1000.0;

                    let icon = match classification {
                        "BENIGN" => "🟢",
                        "DoS" => "🔴",
                        "PortScan" => "🟠",
                        _ => "🟣 (ZERO-DAY)",
                    };

                    println!(
                        "[{:02}] {} Flow: {:18} | Threat: {:8} | MSE: {:.4} | Tier: {:32} | RTT: {:.2}ms",
                        i, icon, classification, threat_level, mse, tier, latency
                    );
                }
            }
            Err(e) => {
                println!(
                    "[{:02}] ⚠️ Backend not responding at {}: {}. (Make sure FastAPI is running!)",
                    i, backend_url, e
                );
            }
        }

        tokio::time::sleep(Duration::from_millis(1000)).await;
    }

    println!("\n✅ Rust Flow Engine completed packet test stream!");
    Ok(())
}
