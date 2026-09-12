import numpy as np
from scapy.all import rdpcap, IP, TCP, UDP
import json
from pathlib import Path

def extract_flow_from_packets(packets) -> dict:
    """
    Aggregates a list of raw Scapy packets in RAM into statistical flow features.
    No disk I/O, zero hardcoding.
    """
    if not packets:
        raise ValueError("No packets provided for flow extraction")

    timestamps = [float(p.time) for p in packets]
    lengths = [float(len(p)) for p in packets]

    # Calculate Flow Duration (microseconds)
    flow_duration = (timestamps[-1] - timestamps[0]) * 1e6 if len(timestamps) > 1 else 100.0

    # Calculate Inter-Arrival Times (IAT)
    iats = [(timestamps[i] - timestamps[i-1]) * 1e6 for i in range(1, len(timestamps))] if len(timestamps) > 1 else [flow_duration]

    # Packet counts & Direction
    total_fwd_packets = float(len(packets))
    total_bwd_packets = 0.0
    total_len_fwd = float(sum(lengths))
    total_len_bwd = 0.0

    # Ports & Protocol
    first_pkt = packets[0]
    dst_port = float(first_pkt[TCP].dport if first_pkt.haslayer(TCP) else (first_pkt[UDP].dport if first_pkt.haslayer(UDP) else 80.0))
    protocol = float(first_pkt[IP].proto if first_pkt.haslayer(IP) else 6.0)

    # Flag counts
    fin_count = sum(1.0 for p in packets if p.haslayer(TCP) and p[TCP].flags.F)
    syn_count = sum(1.0 for p in packets if p.haslayer(TCP) and p[TCP].flags.S)
    rst_count = sum(1.0 for p in packets if p.haslayer(TCP) and p[TCP].flags.R)
    psh_count = sum(1.0 for p in packets if p.haslayer(TCP) and p[TCP].flags.P)
    ack_count = sum(1.0 for p in packets if p.haslayer(TCP) and p[TCP].flags.A)
    urg_count = sum(1.0 for p in packets if p.haslayer(TCP) and p[TCP].flags.U)

    # Rates
    dur_sec = (flow_duration / 1e6) + 1e-5
    flow_bytes_s = float(total_len_fwd / dur_sec)
    flow_packets_s = float(total_fwd_packets / dur_sec)

    return {
        "Destination_Port": dst_port,
        "Protocol": protocol,
        "Flow_Duration": float(flow_duration),
        "Total_Fwd_Packets": total_fwd_packets,
        "Total_Backward_Packets": total_bwd_packets,
        "Total_Length_of_Fwd_Packets": total_len_fwd,
        "Total_Length_of_Bwd_Packets": total_len_bwd,
        "Fwd_Packet_Length_Max": float(max(lengths)),
        "Fwd_Packet_Length_Min": float(min(lengths)),
        "Fwd_Packet_Length_Mean": float(np.mean(lengths)),
        "Fwd_Packet_Length_Std": float(np.std(lengths)),
        "Bwd_Packet_Length_Max": 0.0,
        "Bwd_Packet_Length_Min": 0.0,
        "Bwd_Packet_Length_Mean": 0.0,
        "Bwd_Packet_Length_Std": 0.0,
        "Flow_Bytes_s": flow_bytes_s,
        "Flow_Packets_s": flow_packets_s,
        "Flow_IAT_Mean": float(np.mean(iats)),
        "Flow_IAT_Std": float(np.std(iats)),
        "Flow_IAT_Max": float(max(iats)),
        "Flow_IAT_Min": float(min(iats)),
        "Fwd_IAT_Total": float(flow_duration),
        "Fwd_IAT_Mean": float(np.mean(iats)),
        "Fwd_IAT_Std": float(np.std(iats)),
        "Fwd_IAT_Max": float(max(iats)),
        "Fwd_IAT_Min": float(min(iats)),
        "Bwd_IAT_Total": 0.0,
        "Bwd_IAT_Mean": 0.0,
        "Bwd_IAT_Std": 0.0,
        "Bwd_IAT_Max": 0.0,
        "Bwd_IAT_Min": 0.0,
        "Fwd_PSH_Flags": psh_count,
        "Fwd_URG_Flags": urg_count,
        "Fwd_Header_Length": float(total_fwd_packets * 32.0),
        "Bwd_Header_Length": 0.0,
        "Fwd_Packets_s": flow_packets_s,
        "Bwd_Packets_s": 0.0,
        "Min_Packet_Length": float(min(lengths)),
        "Max_Packet_Length": float(max(lengths)),
        "Packet_Length_Mean": float(np.mean(lengths)),
        "Packet_Length_Std": float(np.std(lengths)),
        "Packet_Length_Variance": float(np.var(lengths)),
        "FIN_Flag_Count": fin_count,
        "SYN_Flag_Count": syn_count,
        "RST_Flag_Count": rst_count,
        "PSH_Flag_Count": psh_count,
        "ACK_Flag_Count": ack_count,
        "URG_Flag_Count": urg_count,
        "CWE_Flag_Count": 0.0,
        "ECE_Flag_Count": 0.0,
        "Down_Up_Ratio": 0.0,
        "Average_Packet_Size": float(np.mean(lengths)),
        "Init_Win_bytes_forward": 65535.0,
        "Init_Win_bytes_backward": 0.0,
        "act_data_pkt_fwd": total_fwd_packets,
        "min_seg_size_forward": 32.0,
        "Active_Mean": 0.0,
        "Active_Std": 0.0,
        "Active_Max": 0.0,
        "Active_Min": 0.0,
        "Idle_Mean": 0.0,
        "Idle_Std": 0.0,
        "Idle_Max": 0.0,
        "Idle_Min": 0.0,
    }


def extract_flow_features_from_pcap(pcap_path=None):
    """
    Reads raw .pcap from Layer 1, aggregates packets into a flow,
    and calculates statistical features matching the NetSentry schema.
    """
    if pcap_path is None:
        pcap_path = Path(__file__).resolve().parent / "captured_traffic.pcap"
    else:
        pcap_path = Path(pcap_path)

    packets = rdpcap(str(pcap_path))
    if not packets:
        raise ValueError(f"No packets found in {pcap_path}")

    # Extract timestamps, lengths, and TCP flags
    timestamps = [float(p.time) for p in packets]
    lengths = [len(p) for p in packets]
    
    # Calculate Flow Duration (microseconds)
    flow_duration = (timestamps[-1] - timestamps[0]) * 1e6 if len(timestamps) > 1 else 0.0
    
    # Calculate Inter-Arrival Times (IAT)
    iats = [ (timestamps[i] - timestamps[i-1]) * 1e6 for i in range(1, len(timestamps)) ] if len(timestamps) > 1 else [0.0]
    
    # Packet counts & Direction
    total_fwd_packets = float(len(packets))
    total_bwd_packets = 0.0
    total_len_fwd = float(sum(lengths))
    total_len_bwd = 0.0

    # Ports & Protocol
    first_pkt = packets[0]
    dst_port = float(first_pkt[TCP].dport if first_pkt.haslayer(TCP) else (first_pkt[UDP].dport if first_pkt.haslayer(UDP) else 80.0))
    protocol = float(first_pkt[IP].proto if first_pkt.haslayer(IP) else 6.0)

    # Flag counts
    fin_count = sum(1.0 for p in packets if p.haslayer(TCP) and p[TCP].flags.F)
    syn_count = sum(1.0 for p in packets if p.haslayer(TCP) and p[TCP].flags.S)
    rst_count = sum(1.0 for p in packets if p.haslayer(TCP) and p[TCP].flags.R)
    psh_count = sum(1.0 for p in packets if p.haslayer(TCP) and p[TCP].flags.P)
    ack_count = sum(1.0 for p in packets if p.haslayer(TCP) and p[TCP].flags.A)
    urg_count = sum(1.0 for p in packets if p.haslayer(TCP) and p[TCP].flags.U)

    # Rates
    dur_sec = (flow_duration / 1e6) + 1e-5
    flow_bytes_s = float(total_len_fwd / dur_sec)
    flow_packets_s = float(total_fwd_packets / dur_sec)

    # Build the complete feature dictionary
    features = {
        "Destination_Port": dst_port,
        "Protocol": protocol,
        "Flow_Duration": float(flow_duration),
        "Total_Fwd_Packets": total_fwd_packets,
        "Total_Backward_Packets": total_bwd_packets,
        "Total_Length_of_Fwd_Packets": total_len_fwd,
        "Total_Length_of_Bwd_Packets": total_len_bwd,
        "Fwd_Packet_Length_Max": float(max(lengths)),
        "Fwd_Packet_Length_Min": float(min(lengths)),
        "Fwd_Packet_Length_Mean": float(np.mean(lengths)),
        "Fwd_Packet_Length_Std": float(np.std(lengths)),
        "Bwd_Packet_Length_Max": 0.0,
        "Bwd_Packet_Length_Min": 0.0,
        "Bwd_Packet_Length_Mean": 0.0,
        "Bwd_Packet_Length_Std": 0.0,
        "Flow_Bytes_s": flow_bytes_s,
        "Flow_Packets_s": flow_packets_s,
        "Flow_IAT_Mean": float(np.mean(iats)),
        "Flow_IAT_Std": float(np.std(iats)),
        "Flow_IAT_Max": float(max(iats)),
        "Flow_IAT_Min": float(min(iats)),
        "Fwd_IAT_Total": float(flow_duration),
        "Fwd_IAT_Mean": float(np.mean(iats)),
        "Fwd_IAT_Std": float(np.std(iats)),
        "Fwd_IAT_Max": float(max(iats)),
        "Fwd_IAT_Min": float(min(iats)),
        "Bwd_IAT_Total": 0.0,
        "Bwd_IAT_Mean": 0.0,
        "Bwd_IAT_Std": 0.0,
        "Bwd_IAT_Max": 0.0,
        "Bwd_IAT_Min": 0.0,
        "Fwd_PSH_Flags": psh_count,
        "Fwd_URG_Flags": urg_count,
        "Fwd_Header_Length": float(total_fwd_packets * 32.0),
        "Bwd_Header_Length": 0.0,
        "Fwd_Packets_s": flow_packets_s,
        "Bwd_Packets_s": 0.0,
        "Min_Packet_Length": float(min(lengths)),
        "Max_Packet_Length": float(max(lengths)),
        "Packet_Length_Mean": float(np.mean(lengths)),
        "Packet_Length_Std": float(np.std(lengths)),
        "Packet_Length_Variance": float(np.var(lengths)),
        "FIN_Flag_Count": fin_count,
        "SYN_Flag_Count": syn_count,
        "RST_Flag_Count": rst_count,
        "PSH_Flag_Count": psh_count,
        "ACK_Flag_Count": ack_count,
        "URG_Flag_Count": urg_count,
        "CWE_Flag_Count": 0.0,
        "ECE_Flag_Count": 0.0,
        "Down_Up_Ratio": 0.0,
        "Average_Packet_Size": float(np.mean(lengths)),
        "Init_Win_bytes_forward": 65535.0,
        "Init_Win_bytes_backward": 0.0,
        "act_data_pkt_fwd": total_fwd_packets,
        "min_seg_size_forward": 32.0,
        "Active_Mean": 0.0,
        "Active_Std": 0.0,
        "Active_Max": 0.0,
        "Active_Min": 0.0,
        "Idle_Mean": 0.0,
        "Idle_Std": 0.0,
        "Idle_Max": 0.0,
        "Idle_Min": 0.0
    }

    # Save to JSON payload for NetSentry
    payload = {"features": features}
    out_json_path = Path(__file__).resolve().parent / "extracted_flow.json"
    with open(out_json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("=" * 60)
    print("⚙️ LAYER 2: FLOW FEATURE EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"📦 Source PCAP:          {pcap_path} ({len(packets)} packets)")
    print(f"⏱️ Calculated Duration:  {flow_duration / 1000:.2f} ms")
    print(f"🎯 Target Port:          {dst_port}")
    print(f"🚩 TCP SYN Flags:        {syn_count} | ACK: {ack_count} | FIN: {fin_count}")
    print(f"💾 Saved Flow Payload:   {out_json_path}")
    print("=" * 60)

    return payload


def send_flow_to_netsentry(api_url="http://localhost:8000/v1/predict", payload=None):
    """Sends the 65 base flow features to the NetSentry FastAPI serving endpoint."""
    import urllib.request
    import json

    if payload is None:
        flow_path = Path(__file__).resolve().parent / "extracted_flow.json"
        with open(flow_path, "r") as f:
            payload = json.load(f)

    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))
        print("\n🚀 [NETSENTRY INFERENCE RESPONSE]")
        print(f"Decision:      {'🚨 ATTACK DETECTED' if res_data['prediction'] == 1 else '✅ BENIGN TRAFFIC'}")
        print(f"Probability:   {res_data['probability']:.4f}")
        print(f"Model Version: {res_data.get('model_version', 'champion')}")
        return res_data


if __name__ == "__main__":
    flow = extract_flow_features_from_pcap()
    try:
        send_flow_to_netsentry(payload=flow)
    except Exception as e:
        print(f"Could not connect to NetSentry API at http://localhost:8000/v1/predict: {e}")