import os
import csv
import h5py
import numpy as np



# Assuming all images are in the same directory
image_dir = '/Users/dk/Desktop/Folders/WashU2024/2024 Fall/CSE659A/project/code/project/images'

# List to store image paths and coordinates
image_paths = []
coordinates = []

# Parse filenames
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg'):
        parts = filename.split('_')
        
        # Check if filename has the expected number of parts
        if len(parts) >= 4:
            try:
                # Parse latitude and longitude
                lat = float(parts[2])
                lon = float(parts[3].replace('.jpg', ''))
                
                # Store image path and coordinates
                image_paths.append(os.path.join(image_dir, filename))
                coordinates.append((lat, lon))
            except ValueError:
                print(f"Skipping file with unexpected format: {filename}")
        else:
            print(f"Skipping file with unexpected format: {filename}")

# Print the first few entries to verify
print(image_paths[:5])
print(coordinates[:5])


# Path to save the HDF5 file
hdf5_path = '/Users/dk/Desktop/Folders/WashU2024/2024 Fall/CSE659A/project/code/project/image_data.h5'

# Create HDF5 file
with h5py.File(hdf5_path, 'w') as hdf:
    # Create datasets for image paths, latitudes, and longitudes
    hdf.create_dataset("image_paths", data=np.string_(image_paths))  # Store image paths as strings
    hdf.create_dataset("latitudes", data=[coord[0] for coord in coordinates])
    hdf.create_dataset("longitudes", data=[coord[1] for coord in coordinates])

print(f"Data saved to {hdf5_path}")




# Path to save the CSV file
# csv_path = '/Users/dk/Desktop/Folders/WashU2024/2024 Fall/CSE659A/project/code/project/image_data.csv'

# # Write data to CSV
# with open(csv_path, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["image_path", "latitude", "longitude"])  # Header

#     for img_path, coord in zip(image_paths, coordinates):
#         writer.writerow([img_path, coord[0], coord[1]])

# print(f"Data saved to {csv_path}")