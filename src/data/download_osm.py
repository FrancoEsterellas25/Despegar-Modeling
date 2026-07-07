import os
import requests
import pandas as pd

def download_osm_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "external")
    os.makedirs(out_dir, exist_ok=True)
    
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    headers = {
        'User-Agent': 'Despegar-Modeling/1.0',
        'Accept': 'application/json'
    }
    
    # ---------------------------------------------------------
    # 1. Descargar Estaciones de Metro
    # ---------------------------------------------------------
    print("Descargando estaciones de metro de Rio de Janeiro desde OSM...")
    metro_query = """
    [out:json][timeout:25];
    area[name="Rio de Janeiro"]->.searchArea;
    node["station"="subway"](area.searchArea);
    out body;
    """
    
    response = requests.get(overpass_url, headers=headers, params={'data': metro_query})
    if response.status_code == 200:
        data = response.json()
        metro_coords = []
        for element in data['elements']:
            if 'lat' in element and 'lon' in element:
                metro_coords.append({'lat': element['lat'], 'lon': element['lon']})
        
        df_metro = pd.DataFrame(metro_coords)
        if not df_metro.empty:
            metro_path = os.path.join(out_dir, "metro_estaciones_cache.csv")
            df_metro.to_csv(metro_path, index=False)
            print(f"EXITO: Metro guardado: {len(df_metro)} estaciones en {metro_path}")
        else:
            print("AVISO: No se encontraron estaciones de metro en la respuesta.")
    else:
        print(f"ERROR: Error al consultar metro: HTTP {response.status_code}")
        print(response.text)

    # ---------------------------------------------------------
    # 2. Descargar Línea de Costa
    # ---------------------------------------------------------
    print("\nDescargando línea de costa de Rio de Janeiro desde OSM...")
    costa_query = """
    [out:json][timeout:50];
    area[name="Rio de Janeiro"]->.searchArea;
    way["natural"="coastline"](area.searchArea);
    out geom;
    """
    
    response = requests.get(overpass_url, headers=headers, params={'data': costa_query})
    if response.status_code == 200:
        data = response.json()
        costa_coords = []
        for element in data['elements']:
            if 'geometry' in element:
                for point in element['geometry']:
                    costa_coords.append({'lat': point['lat'], 'lon': point['lon']})
        
        df_costa = pd.DataFrame(costa_coords)
        if not df_costa.empty:
            costa_path = os.path.join(out_dir, "costa_linea_cache.csv")
            df_costa.to_csv(costa_path, index=False)
            print(f"EXITO: Costa guardada: {len(df_costa)} puntos geograficos en {costa_path}")
        else:
            print("AVISO: No se encontraron puntos de costa en la respuesta.")
    else:
        print(f"ERROR: Error al consultar costa: HTTP {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    download_osm_data()
