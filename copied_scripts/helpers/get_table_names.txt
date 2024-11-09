import os

# Replace with your folder path
folder_path = r'C:/Users/ZachDesktop/Documents/GitHub/BD4H/project/draft/mimic-iii-clinical-database-demo-1.4'

# List all files in the specified folder
file_names = os.listdir(folder_path)

# Print each file name
for file_name in file_names:
    print(file_name)
