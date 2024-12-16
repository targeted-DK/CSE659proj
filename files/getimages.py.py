import random
import requests
from PIL import Image
from io import BytesIO
from city_bounds import city_bounds
from dotenv import load_dotenv
import os

load_dotenv() 
API_KEY = os.getenv("API_KEY")


def get_random_coordinates(bounds):
    lat = random.uniform(bounds['south'], bounds['north'])
    lng = random.uniform(bounds['west'], bounds['east'])
    return lat, lng

def request_street_view_image(lat, lng):
    metadata_url = f"https://maps.googleapis.com/maps/api/streetview/metadata?location={lat},{lng}&key={API_KEY}"
    metadata_response = requests.get(metadata_url).json()

    if metadata_response['status'] == 'OK':
        street_view_url = f"https://maps.googleapis.com/maps/api/streetview?size=640x640&location={lat},{lng}&fov=90&heading=0&pitch=0&key={API_KEY}"
        return street_view_url
    else:
        print(f"No street view available for coordinates: {lat}, {lng}")
        return None

def fetch_and_save_street_view_image(lat, lng, count, file_name="street_view_image.jpg"):
    image_url = request_street_view_image(lat, lng)
    
    if image_url:
        response = requests.get(image_url)
        
        if response.status_code == 200:
            # Open and resize the image to 224x224
            img = Image.open(BytesIO(response.content))
            
            count = count+50
            file_location = f"../images/test/capitals/{file_name}_{count}_{lat}_{lng}.jpg"
            img.save(file_location)
            print(f"Image saved as images/test/capitals/{file_name}_{count}_{lat}_{lng}.jpg")

            return True
        else:
            print(f"Failed to retrieve image, status code: {response.status_code}")
            return False
    else:
        print("No image URL available for the given coordinates")
        return False

for city, bounds in city_bounds.items():
    print(f"Fetching images for {city}...")
    successful_images = 0
    attempts = 0
    
    while successful_images < 5:
        attempts += 1
        lat, lng = get_random_coordinates(bounds)
        success = fetch_and_save_street_view_image(lat, lng, successful_images, city)
        
        if success: 
            successful_images += 1 
        
       
        if attempts >= 20:
            print(f"Stopped after 100 attempts for {city}")
            break
            
    
    print(f"Total successful images for {city}: {successful_images}")