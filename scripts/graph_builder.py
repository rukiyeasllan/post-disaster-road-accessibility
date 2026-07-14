import cv2
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import os


def predict_mask(model, image_path, device, img_size=512):
    from PIL import Image
    import torch
    image = Image.open(image_path).convert("RGB")
    orig_size = image.size
    image = image.resize((img_size, img_size), Image.BILINEAR)
    image = np.array(image, dtype=np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    image = (image - mean) / std
    image = image.unsqueeze(0).to(device)
    import torch.nn.functional as F
    with torch.no_grad():
        output = model(image)
        pred = (torch.sigmoid(output) > 0.5).float()
    pred = pred.squeeze().cpu().numpy().astype(np.uint8)
    pred = cv2.resize(pred, orig_size, interpolation=cv2.INTER_NEAREST)
    return pred


def clean_mask(mask):
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def mask_to_skeleton(mask):
    mask_bool = mask.astype(bool)
    skeleton = skeletonize(mask_bool)
    return skeleton.astype(np.uint8)


def skeleton_to_graph(skeleton):
    G = nx.Graph()
    h, w = skeleton.shape
    coords = np.argwhere(skeleton > 0)
    for y, x in coords:
        G.add_node((y, x))
    for y, x in coords:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx_ < w and skeleton[ny, nx_] > 0:
                    dist = np.sqrt(dy**2 + dx**2)
                    G.add_edge((y, x), (ny, nx_), weight=dist)
    return G


def find_junction_nodes(G):
    junctions = [node for node in G.nodes() if G.degree(node) >= 3]
    endpoints = [node for node in G.nodes() if G.degree(node) == 1]
    return junctions, endpoints


def simplify_graph(G, junctions, endpoints, sample_interval=5):
    key_nodes = set(junctions + endpoints)
    # Her segment uzerinde aralik noktalari ekle
    visited_edges = set()
    for start in list(key_nodes):
        for neighbor in G.neighbors(start):
            if (start, neighbor) in visited_edges:
                continue
            path = [start, neighbor]
            prev = start
            curr = neighbor
            total_weight = G[start][neighbor].get("weight", 1.0)
            step = 0
            while curr not in key_nodes:
                nexts = [n for n in G.neighbors(curr) if n != prev]
                if not nexts:
                    break
                step += 1
                if step % sample_interval == 0:
                    key_nodes.add(curr)
                    break
                nxt = nexts[0]
                total_weight += G[curr][nxt].get("weight", 1.0)
                prev, curr = curr, nxt
                path.append(curr)
            visited_edges.add((start, curr))
            visited_edges.add((curr, start))
    simplified = nx.Graph()
    simplified.add_nodes_from(key_nodes)
    for start in key_nodes:
        for neighbor in G.neighbors(start):
            path = [start, neighbor]
            prev = start
            curr = neighbor
            total_weight = G[start][neighbor].get("weight", 1.0)
            while curr not in key_nodes:
                nexts = [n for n in G.neighbors(curr) if n != prev]
                if not nexts:
                    break
                nxt = nexts[0]
                total_weight += G[curr][nxt].get("weight", 1.0)
                prev, curr = curr, nxt
                path.append(curr)
            if curr in key_nodes and curr != start:
                if not simplified.has_edge(start, curr):
                    simplified.add_edge(start, curr, weight=total_weight, path=path)
    return simplified


def visualize_graph(image_path, mask, skeleton, graph, output_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image)
    axes[0].set_title("Satellite Image", fontsize=22, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Road Mask", fontsize=22, fontweight="bold")
    axes[1].axis("off")
    kernel = np.ones((4, 4), np.uint8)
    skeleton_vis = cv2.dilate(skeleton, kernel, iterations=2)
    axes[2].imshow(skeleton_vis, cmap="gray")
    axes[2].set_title("Skeleton", fontsize=22, fontweight="bold")
    axes[2].axis("off")
    overlay = image.copy()
    for node in graph.nodes():
        y, x = node
        if y < overlay.shape[0] and x < overlay.shape[1]:
            cv2.circle(overlay, (x, y), 3, (0, 255, 0), -1)
    for u, v in graph.edges():
        y1, x1 = u
        y2, x2 = v
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 1)
    axes[3].imshow(overlay)
    axes[3].set_title("Yol Grafi")
    axes[3].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Kaydedildi: {output_path}")


def process_image(image_path, model, device, output_dir, img_size=512):
    os.makedirs(output_dir, exist_ok=True)
    mask = predict_mask(model, image_path, device, img_size)
    mask = clean_mask(mask)
    skeleton = mask_to_skeleton(mask)
    G_full = skeleton_to_graph(skeleton)
    junctions, endpoints = find_junction_nodes(G_full)
    # Her 10 pikselde bir dugum al
    all_nodes = list(G_full.nodes())
    sampled = set(all_nodes[::10])
    sampled.update(junctions)
    sampled.update(endpoints)
    G_simplified = G_full.subgraph(sampled).copy()
    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{basename}_graph.png")
    visualize_graph(image_path, mask, skeleton, G_simplified, output_path)
    print(f"Dugum sayisi: {G_simplified.number_of_nodes()}")
    print(f"Kenar sayisi: {G_simplified.number_of_edges()}")
    return mask, skeleton, G_simplified
