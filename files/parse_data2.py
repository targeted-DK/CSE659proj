import os
import csv
import h5py
import numpy as np
import geohash2

image_dir = r"C:\Users\mysoo\OneDrive\Documents\GitHub\CSE659proj\images"

image_paths = []
prompts = []
geohash_coordinates = []

for filename in os.listdir(image_dir):
    if filename.endswith('.jpg'):
        parts = filename.split('_')
        
        if len(parts) == 4:
            try:
                lat = float(parts[2])
                lon = float(parts[3].replace('.jpg', ''))
                geohash_code = geohash2.encode(lat, lon, precision=2)
                
                image_paths.append(os.path.join(image_dir, filename))
              
                prompts.append("A street in " + parts[0])
              
            
                geohash_coordinates.append(geohash_code)
            except ValueError:
                print(f"Skipping file with unexpected format: {filename}")
                
        elif len(parts) == 5:
            try:
                lat = float(parts[3])
                lon = float(parts[4].replace('.jpg', ''))
                geohash_code = geohash2.encode(lat, lon, precision=2)
                image_paths.append(os.path.join(image_dir, filename))
         
                prompts.append("A street in " + parts[0] + parts[1])
                
                geohash_coordinates.append(geohash_code)
            except ValueError:
                print(f"Skipping file with unexpected format: {filename}")
                
        elif len(parts) == 6:
            try:
                lat = float(parts[4])
                lon = float(parts[5].replace('.jpg', ''))
                geohash_code = geohash2.encode(lat, lon, precision=2)
                image_paths.append(os.path.join(image_dir, filename))
            
                prompts.append("A street in " + parts[0] + parts[1] + parts[2])
                
                geohash_coordinates.append(geohash_code)
            except ValueError:
                print(f"Skipping file with unexpected format: {filename}")
                
        else:
            print(f"Skipping file with unexpected format: {filename}")

print(image_paths[:5])
print(geohash_coordinates[:5])



# hdf5_path = r"C:\Users\mysoo\OneDrive\Documents\GitHub\CSE659proj\image_data.h5"

# with h5py.File(hdf5_path, 'w') as hdf:
#     # Create datasets for image paths and geohashes
#     hdf.create_dataset("image_paths", data=np.bytes_(image_paths))  # Store image paths as strings
#     hdf.create_dataset("geohashes", data=np.bytes_(geohash_coordinates))  # Store geohashes as strings

# print(f"Data saved to {hdf5_path}")

# Path to save the CSV file
csv_path = r'C:\Users\mysoo\OneDrive\Documents\GitHub\CSE659proj\image_data2.csv'


with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["image_path","prompt", "geohash"])  # Header

    for img_path, prompt, geohash in zip(image_paths, prompts, geohash_coordinates):
        writer.writerow([img_path, prompt, geohash])

print(f"Data saved to {csv_path}")