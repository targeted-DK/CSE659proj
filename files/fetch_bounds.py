import requests

capital_cities = [
    "Kabul", "Tirana", "Algiers", "Andorra la Vella", "Luanda", "Saint John's", 
    "Buenos Aires", "Yerevan", "Canberra", "Vienna", "Baku", "Nassau", "Manama", 
    "Dhaka", "Bridgetown", "Minsk", "Brussels", "Belmopan", "Porto-Novo", 
    "Thimphu", "Sucre", "Sarajevo", "Gaborone", "Brasília", "Bandar Seri Begawan", 
    "Sofia", "Ouagadougou", "Gitega", "Phnom Penh", "Yaoundé", "Ottawa", "Praia", 
    "Bangui", "N'Djamena", "Santiago", "Beijing", "Bogotá", "Moroni", 
    "Brazzaville", "San José", "Zagreb", "Havana", "Nicosia", "Prague", 
    "Copenhagen", "Djibouti", "Roseau", "Santo Domingo", "Quito", "Cairo", 
    "San Salvador", "Malabo", "Asmara", "Tallinn", "Mbabane", "Addis Ababa", 
    "Suva", "Helsinki", "Paris", "Libreville", "Banjul", "Tbilisi", "Berlin", 
    "Accra", "Athens", "Saint George's", "Guatemala City", "Conakry", "Bissau", 
    "Georgetown", "Port-au-Prince", "Tegucigalpa", "Budapest", "Reykjavík", 
    "New Delhi", "Jakarta", "Tehran", "Baghdad", "Dublin", "Jerusalem", "Rome", 
    "Kingston", "Tokyo", "Amman", "Astana", "Nairobi", "Tarawa", "Pyongyang", 
    "Seoul", "Kuwait City", "Bishkek", "Vientiane", "Riga", "Beirut", "Maseru", 
    "Monrovia", "Tripoli", "Vaduz", "Vilnius", "Luxembourg City", 
    "Antananarivo", "Lilongwe", "Kuala Lumpur", "Malé", "Bamako", "Valletta", 
    "Majuro", "Nouakchott", "Port Louis", "Mexico City", "Palikir", "Chișinău", 
    "Monaco", "Ulaanbaatar", "Podgorica", "Rabat", "Maputo", "Naypyidaw", 
    "Windhoek", "Yaren", "Kathmandu", "Amsterdam", "Wellington", "Managua", 
    "Niamey", "Abuja", "Oslo", "Muscat", "Islamabad", "Ngerulmud", 
    "Panama City", "Port Moresby", "Asunción", "Lima", "Manila", "Warsaw", 
    "Lisbon", "Doha", "Bucharest", "Moscow", "Kigali", "Basseterre", "Castries", 
    "Kingstown", "Apia", "San Marino", "Riyadh", "Dakar", "Belgrade", "Victoria", 
    "Freetown", "Singapore", "Bratislava", "Ljubljana", "Honiara", "Mogadishu", 
    "Pretoria", "Juba", "Madrid", "Sri Jayawardenepura Kotte", "Khartoum", 
    "Paramaribo", "Stockholm", "Bern", "Damascus", "Taipei", "Dushanbe", 
    "Dodoma", "Bangkok", "Dili", "Lomé", "Nukuʻalofa", "Port of Spain", 
    "Tunis", "Ankara", "Ashgabat", "Funafuti", "Kampala", "Kyiv", "Abu Dhabi", 
    "London", "Washington D.C.", "Montevideo", "Tashkent", "Port Vila", 
    "Vatican City", "Caracas", "Hanoi", "Sana'a", "Lusaka", "Harare"
]




def fetch_city_bounds(city_name):
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={city_name}&format=json&polygon_geojson=1"
        headers = {"User-Agent": "CapitalCityBoundsFetcher/1.0"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        if data:
            bbox = data[0]["boundingbox"]
            north, south, west, east = float(bbox[1]), float(bbox[0]), float(bbox[2]), float(bbox[3])
            return {
                "north": round(north, 2),
                "south": round(south, 2),
                "west": round(west, 2),
                "east": round(east, 2)
            }
        else:
            print(f"No data found for {city_name}")
            return None
    except Exception as e:
        print(f"Error fetching data for {city_name}: {e}")
        return None

capital_city_bounds = {}
for city in capital_cities:
    bounds = fetch_city_bounds(city)
    if bounds:
        city_key = city.replace(" ", "_").replace(".", "")
        capital_city_bounds[city_key] = bounds

print(capital_city_bounds)