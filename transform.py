import json
from pyproj import Transformer

# Configuration: Source (WGS84) -> Target (UTM Zone 32N - Standard for Denmark)
# Change "EPSG:25832" to "EPSG:3857" if you need Web Mercator for web maps.
transformer = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)

input_file = 'denmark_ev_data.json'
output_file = 'denmark_ev_data_projected.json'

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        # Access the AddressInfo object where coords are stored
        address_info = item.get('AddressInfo', {})
        
        lat = address_info.get('Latitude')
        lon = address_info.get('Longitude')

        if lat is not None and lon is not None:
            # Transform returns (x, y) because always_xy=True
            x, y = transformer.transform(lon, lat)
            
            # Inject new keys into the AddressInfo object
            address_info['x_utm32'] = x
            address_info['y_utm32'] = y

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        
    print(f"Successfully processed {len(data)} records.")

except Exception as e:
    print(f"Error: {e}")
