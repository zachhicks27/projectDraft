import os
import pandas as pd

# Replace with your folder path
folder_path = r'data/processed'
output_file = 'output_summary2.txt'

# Open the output file for writing
with open(output_file, 'w') as f:
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            
            # Read the CSV file
            try:
                df = pd.read_csv(file_path)
                
                # Write filename
                f.write(f"Filename: {file_name}\n")
                
                # Write column names
                f.write("Column Names:\n")
                f.write(", ".join(df.columns) + "\n")
                
                # Write the first row of data (if it exists)
                if not df.empty:
                    f.write("First Row of Data:\n")
                    f.write(", ".join(df.iloc[0].astype(str).values) + "\n")
                
                # Add a separator for clarity between files
                f.write("\n" + "-"*40 + "\n\n")
            
            except Exception as e:
                f.write(f"Error processing {file_name}: {e}\n")
                f.write("\n" + "-"*40 + "\n\n")

print(f"Summary written to {output_file}")
