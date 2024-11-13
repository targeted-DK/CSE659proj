import geohash2

# Encode coordinates into Geohash
latitude, longitude = 37.7749, -122.4194
geohash_code = geohash2.encode(latitude, longitude, precision=7)

# Decode Geohash back to approximate latitude and longitude
lat, lon = geohash2.decode(geohash_code)
print(f"Original coordinates: ({latitude}, {longitude})")
print(f"Decoded coordinates from Geohash: ({lat}, {lon})")