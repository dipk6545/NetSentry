from pathlib import Path
import time
from scapy.all import sniff, wrpcap, IP

def packet_callback(packet):
    if packet.haslayer(IP):
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        protocol = packet[IP].proto
        print(f"📦 [{time.strftime('%H:%M:%S')}] {src_ip} ➔ {dst_ip} (Proto: {protocol})")

def start_capture(packet_count=10, output_file=None):
    if output_file is None:
        output_file = str(Path(__file__).resolve().parent / "captured_traffic.pcap")
    print(f"🚀 Starting Layer 1 Packet Sniffer... Capturing {packet_count} live packets...")
    
    # Sniff live packets from your network interface
    packets = sniff(count=packet_count, prn=packet_callback)
    
    # Save to standard .pcap format
    wrpcap(output_file, packets)
    print(f"\n💾 Saved {len(packets)} raw packets to {output_file}!")

if __name__ == "__main__":
    start_capture(packet_count=10)