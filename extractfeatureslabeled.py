import os
import pandas as pd
from tqdm import tqdm
from scapy.all import rdpcap, IP

dataset_dir = r"D:\nias\datasets"

all_features = []

def extract_features(packets, file_name):
    features = []
    class_label = "vpn" if file_name.lower().startswith("vpn") else "nonvpn"
    
    for pkt in packets:
        if IP in pkt:
            ip_layer = pkt[IP]
            feature = {
                "filename": file_name,
                "packet_length": len(pkt),
                "protocol": ip_layer.proto,
                "src_ip": ip_layer.src,
                "dst_ip": ip_layer.dst,
                "class_label": class_label  # Add VPN or Non-VPN label
            }
            features.append(feature)
    return features

for root, dirs, files in os.walk(dataset_dir):
    for file in tqdm(files):
        if file.endswith(".pcap"):
            file_path = os.path.join(root, file)
            print(f"Processing: {file_path}")
            try:
                packets = rdpcap(file_path)
                print(f"Extracted {len(packets)} packets from {file}")
                extracted = extract_features(packets, file)
                if extracted:
                    all_features.append(pd.DataFrame(extracted))
                else:
                    print(f"No IP packets found in {file}")
            except Exception as e:
                print(f"Error reading {file}: {e}")

# Save features to CSV
if all_features:
    final_df = pd.concat(all_features, ignore_index=True)
    final_df.to_csv("extracted_features_labeled.csv", index=False)
    print("Feature extraction complete. Saved to extracted_features_labeled.csv")
else:
    print("No features extracted. Ensure packets contain IP layer data.")
