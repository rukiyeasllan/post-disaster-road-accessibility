import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import os

HIGHWAY_PRIORITY = {
    "motorway": 5, "trunk": 5, "primary": 4,
    "secondary": 3, "tertiary": 2, "residential": 1,
    "unclassified": 1, "service": 1
}

def get_osm_graph(lat, lon, dist=500):
    print(f"OSM verisi indiriliyor: ({lat}, {lon}), {dist}m radius...")
    G = ox.graph_from_point((lat, lon), dist=dist, network_type="drive")
    G = ox.project_graph(G)
    print(f"OSM dugum: {G.number_of_nodes()}, kenar: {G.number_of_edges()}")
    return G

def get_road_priority(highway_type):
    if isinstance(highway_type, list):
        highway_type = highway_type[0]
    return HIGHWAY_PRIORITY.get(highway_type, 1)

def osm_to_edges(G_osm):
    edges = []
    for u, v, data in G_osm.edges(data=True):
        highway = data.get("highway", "unclassified")
        priority = get_road_priority(highway)
        if "geometry" in data:
            edges.append({"geometry": data["geometry"], "highway": highway, "priority": priority})
        else:
            u_data = G_osm.nodes[u]
            v_data = G_osm.nodes[v]
            line = LineString([(u_data["x"], u_data["y"]), (v_data["x"], v_data["y"])])
            edges.append({"geometry": line, "highway": highway, "priority": priority})
    return edges

def pixel_graph_to_geo(G_pixel, bounds, img_shape, heading=0.0):
    import math
    h, w = img_shape
    min_lon, min_lat, max_lon, max_lat = bounds
    cx_geo = (min_lon + max_lon) / 2
    cy_geo = (min_lat + max_lat) / 2
    cos_h = math.cos(math.radians(-heading))
    sin_h = math.sin(math.radians(-heading))
    G_geo = nx.Graph()
    for node in G_pixel.nodes():
        y_px, x_px = node
        # Piksel -> normalize [-0.5, 0.5]
        nx_ = (x_px / w) - 0.5
        ny_ = 0.5 - (y_px / h)
        # Ters rotasyon uygula
        nx_r = cos_h * nx_ - sin_h * ny_
        ny_r = sin_h * nx_ + cos_h * ny_
        # Geo koordinata cevir
        lon = cx_geo + nx_r * (max_lon - min_lon)
        lat = cy_geo + ny_r * (max_lat - min_lat)
        G_geo.add_node(node, lon=lon, lat=lat)
    for u, v, data in G_pixel.edges(data=True):
        G_geo.add_edge(u, v, **data)
    return G_geo

def compare_graphs(G_post_pixel, G_osm, bounds, img_shape, delta=50, heading=0.0, mask=None):
    """OSM kenarlarini dogrudan yol maskesi uzerinde kontrol eder."""
    import pyproj, math
    edges = osm_to_edges(G_osm)
    h, w = img_shape
    min_lon, min_lat, max_lon, max_lat = bounds
    proj = pyproj.Proj(G_osm.graph["crs"])
    cos_h = math.cos(math.radians(heading))
    sin_h = math.sin(math.radians(heading))
    cx_geo = (min_lon + max_lon) / 2
    cy_geo = (min_lat + max_lat) / 2

    def geo_to_px(lon, lat):
        dlat = lat - cy_geo
        dlon = lon - cx_geo
        dlat_r = cos_h * dlat - sin_h * dlon
        dlon_r = sin_h * dlat + cos_h * dlon
        x = int(round((dlon_r + (max_lon - min_lon)/2) / (max_lon - min_lon) * w))
        y = int(round(((max_lat - min_lat)/2 - dlat_r) / (max_lat - min_lat) * h))
        return max(0, min(w-1, x)), max(0, min(h-1, y))

    # delta metre -> piksel
    res_m_per_px = (max_lon - min_lon) * 111320 / w
    delta_px = max(1, int(delta / res_m_per_px))

    damaged = []
    safe = []
    for edge in edges:
        coords = list(edge["geometry"].coords)
        road_pixels = 0
        total_pixels = 0
        for cx_utm, cy_utm in coords:
            try:
                lon_geo, lat_geo = proj(cx_utm, cy_utm, inverse=True)
            except:
                continue
            x_px, y_px = geo_to_px(lon_geo, lat_geo)
            # delta_px genisliginde maske kontrolu
            x0 = max(0, x_px - delta_px)
            x1 = min(w-1, x_px + delta_px)
            y0 = max(0, y_px - delta_px)
            y1 = min(h-1, y_px + delta_px)
            if mask is not None:
                patch = mask[y0:y1+1, x0:x1+1]
                road_pixels += int(patch.sum())
                total_pixels += patch.size
            else:
                total_pixels += 1
        # En az %20 yol pikseli varsa guvenli
        ratio = road_pixels / total_pixels if total_pixels > 0 else 0
        if ratio >= 0.10:
            safe.append(edge)
        else:
            damaged.append(edge)
    high_priority_damaged = [e for e in damaged if e["priority"] >= 3]
    print(f"Toplam OSM kenar     : {len(edges)}")
    print(f"Guvenli yol          : {len(safe)}")
    print(f"Hasarli/kapali yol   : {len(damaged)}")
    print(f"Yuksek oncelikli hasar: {len(high_priority_damaged)}")
    return safe, damaged

def visualize_damage(image_path, safe_edges, damaged_edges, bounds, img_shape, output_path, osm_crs=None, heading=0.0):
    import cv2
    import pyproj
    image = cv2.imread(image_path)
    image = cv2.resize(image, (img_shape[1], img_shape[0]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = img_shape
    min_lon, min_lat, max_lon, max_lat = bounds

    import math
    cos_h = math.cos(math.radians(heading))
    sin_h = math.sin(math.radians(heading))
    cx_geo = (min_lon + max_lon) / 2
    cy_geo = (min_lat + max_lat) / 2
    def geo_to_px(lon, lat):
        # Merkeze gore fark
        dlat = lat - cy_geo
        dlon = lon - cx_geo
        # Dondurulen koordinat
        dlat_r = cos_h * dlat - sin_h * dlon
        dlon_r = sin_h * dlat + cos_h * dlon
        # Piksel donusumu
        x = int(round((dlon_r + (max_lon - min_lon)/2) / (max_lon - min_lon) * w))
        y = int(round(((max_lat - min_lat)/2 - dlat_r) / (max_lat - min_lat) * h))
        x = max(0, min(w-1, x))
        y = max(0, min(h-1, y))
        return (x, y)

    def proj_to_px(px, py, proj):
        try:
            lon, lat = proj(px, py, inverse=True)
            return geo_to_px(lon, lat)
        except:
            return None

    proj = pyproj.Proj(osm_crs) if osm_crs else None
    overlay = image.copy()

    for edge in safe_edges:
        coords = list(edge["geometry"].coords)
        if proj:
            pts = [proj_to_px(c[0], c[1], proj) for c in coords]
        else:
            pts = [geo_to_px(c[0], c[1]) for c in coords]
        pts = [p for p in pts if p is not None]
        thickness = 1 + edge["priority"]
        for i in range(len(pts)-1):
            cv2.line(overlay, pts[i], pts[i+1], (0, 255, 0), thickness)

    for edge in damaged_edges:
        coords = list(edge["geometry"].coords)
        if proj:
            pts = [proj_to_px(c[0], c[1], proj) for c in coords]
        else:
            pts = [geo_to_px(c[0], c[1]) for c in coords]
        pts = [p for p in pts if p is not None]
        thickness = 1 + edge["priority"]
        color = (255, 0, 0) if edge["priority"] >= 3 else (255, 165, 0)
        for i in range(len(pts)-1):
            cv2.line(overlay, pts[i], pts[i+1], color, thickness)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=22, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title("Damage Assessment", fontsize=22, fontweight="bold")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Hasar haritasi kaydedildi: {output_path}")
