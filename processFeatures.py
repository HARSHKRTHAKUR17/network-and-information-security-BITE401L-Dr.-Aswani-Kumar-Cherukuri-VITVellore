import pandas as pd

# Define the chunk size
chunk_size = 100000  
output_file = "processed_features_labeled.csv"  

# Open the output file in write mode and write the header
first_chunk = True  

for chunk in pd.read_csv("extracted_features_labeled.csv", chunksize=chunk_size):
    print(f"Processing new chunk...")

    # Drop unnecessary columns
    chunk.drop(columns=['filename'], inplace=True)

    # Normalize numerical features
    chunk['packet_length'] = (chunk['packet_length'] - chunk['packet_length'].min()) / (chunk['packet_length'].max() - chunk['packet_length'].min())

    # Convert categorical features to numerical values
    chunk['protocol'] = chunk['protocol'].astype('category').cat.codes
    chunk['src_ip'] = chunk['src_ip'].astype('category').cat.codes
    chunk['dst_ip'] = chunk['dst_ip'].astype('category').cat.codes

    # Save to a new file
    chunk.to_csv(output_file, mode='a', index=False, header=first_chunk)
    first_chunk = False  # Only write the header for the first chunk

print("Processing complete! Processed data saved in 'processed_features_labeled.csv'.")
