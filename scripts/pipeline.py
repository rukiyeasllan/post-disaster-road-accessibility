import os
import sys
import torch
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SegFormerRoad
from graph_builder import process_image
from osm_compare import get_osm_graph, compare_graphs, visualize_damage


def run_pipeline(image_path, lat, lon, model_path, output_dir, osm_dist=500, img_size=512, res=0.30517578125, heading=0.0, delta=15.0):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    # 1. Model yukle
    print("\n[1/4] Model yukleniyor...")
    model = SegFormerRoad(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 2. Maske + iskelet + graf
    print("\n[2/4] Yol maskesi ve graf olusturuluyor...")
    mask, skeleton, G_post = process_image(image_path, model, device, output_dir, img_size)


    # 4. Karsilastir
    print("\n[4/4] Hasar tespiti yapiliyor...")
    # Gercek bounds hesapla
    import rasterio
    from rasterio.warp import transform as rio_transform
    from PIL import Image as PILImage
    img_w, img_h = PILImage.open(image_path).size
    # Piksel basina derece hesapla (yaklasik)
    from pyproj import Transformer
    # res parametre olarak gelir
    half_m_w = (img_w / 2) * res
    half_m_h = (img_h / 2) * res
    utm_zone = int((lon + 180) / 6) + 1
    epsg_utm = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone
    t_fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_utm}", always_xy=True)
    t_inv = Transformer.from_crs(f"EPSG:{epsg_utm}", "EPSG:4326", always_xy=True)
    cx_utm, cy_utm = t_fwd.transform(lon, lat)
    lon_min, lat_min = t_inv.transform(cx_utm - half_m_w, cy_utm - half_m_h)
    lon_max, lat_max = t_inv.transform(cx_utm + half_m_w, cy_utm + half_m_h)
    bounds = (lon_min, lat_min, lon_max, lat_max)
    # OSM dist'i bounds'a gore otomatik ayarla
    bounds_w_m = (lon_max - lon_min) * 111320
    bounds_h_m = (lat_max - lat_min) * 111320
    auto_dist = int(max(bounds_w_m, bounds_h_m) / 2 * 1.2)
    print(f"Bounds: {bounds_w_m:.0f}m x {bounds_h_m:.0f}m -> OSM dist={auto_dist}m")
    G_osm = get_osm_graph(lat, lon, dist=auto_dist)
    img_shape = mask.shape
    safe, damaged = compare_graphs(G_post, G_osm, bounds, img_shape, heading=heading, delta=delta, mask=mask)

    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{basename}_damage_map.png")
    visualize_damage(image_path, safe, damaged, bounds, img_shape, output_path, osm_crs=G_osm.graph.get("crs"), heading=heading)

    print("\n=== SONUCLAR ===")
    print(f"Toplam OSM yol segmenti : {len(safe) + len(damaged)}")
    print(f"Guvenli yol             : {len(safe)}")
    print(f"Hasarli/kapali yol      : {len(damaged)}")
    if len(safe) + len(damaged) > 0:
        acc = len(safe) / (len(safe) + len(damaged)) * 100
        print(f"Erisim orani            : {acc:.1f}%")
    print(f"\nCikti klasoru: {output_dir}")
    return safe, damaged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Afet Sonrasi Yol Hasar Tespiti")
    parser.add_argument("--image", required=True, help="Uydu goruntusunun yolu")
    parser.add_argument("--lat", type=float, required=True, help="Enlem")
    parser.add_argument("--lon", type=float, required=True, help="Boylam")
    parser.add_argument("--model", default=os.path.expanduser("~/road_damage_project/models/best_model.pth"))
    parser.add_argument("--output", default=os.path.expanduser("~/road_damage_project/outputs"))
    parser.add_argument("--dist", type=int, default=500, help="OSM radius (metre)")
    parser.add_argument("--res", type=float, default=0.30517578125, help="GSD m/px")
    parser.add_argument("--img-size", type=int, default=512, dest="img_size", help="Model input size")
    parser.add_argument("--heading", type=float, default=0.0, help="Goruntu kuzey acisi (derece, saat yonu)")
    parser.add_argument("--delta", type=float, default=15.0, help="OSM eslestirme toleransi (metre)")
    args = parser.parse_args()
    run_pipeline(
        image_path=args.image,
        lat=args.lat,
        lon=args.lon,
        model_path=args.model,
        output_dir=args.output,
        osm_dist=args.dist,
        res=args.res,
        img_size=args.img_size,
        heading=args.heading,
        delta=args.delta
    )
