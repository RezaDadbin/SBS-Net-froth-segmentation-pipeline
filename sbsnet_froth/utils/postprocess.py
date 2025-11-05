import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import matplotlib.patches as patches

# Default parameters
MERGED_PEAKS_MIN = 2        # if number of peaks >= this: treat as merged and run watershed
MORPH_KERNEL = (3, 3)       # kernel used for morphological open/close on ROIs
MORPH_OPEN_ITERS = 1
MORPH_CLOSE_ITERS = 1
MIN_AREA = 2                # remove blobs smaller than this area (pixels)
SEED_K_FACTOR = -1.0        # for statistical thresholding
DEBUG_SAVE_DIR = "./debug_roi_watershed"


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def simple_threshold_mask(norm_img: np.ndarray, thresh: float) -> np.ndarray:
    """Return binary uint8 mask (0/255) from normalized image [0..1]."""
    return (norm_img > thresh).astype(np.uint8) * 255


def find_seed_components(roi_mask: np.ndarray, k_factor: float = SEED_K_FACTOR):
    """Distance transform + statistical threshold → seed components."""
    dist = cv2.distanceTransform(roi_mask, cv2.DIST_L1, 3)
    if dist.max() <= 0:
        return np.zeros_like(roi_mask), 0, dist

    internal_distances = dist[roi_mask == 255]
    if internal_distances.size < 1:
        return np.zeros_like(roi_mask), 0, dist

    mu = float(np.mean(internal_distances))
    sigma = float(np.std(internal_distances))
    thresh_val = mu + k_factor * sigma

    _, seed_mask = cv2.threshold(dist, thresh_val, 255, cv2.THRESH_BINARY)
    seed_mask = seed_mask.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL)
    seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    n_labels, _ = cv2.connectedComponents(seed_mask)
    return seed_mask, n_labels, dist


def local_watershed_on_roi(roi_img_norm: np.ndarray, roi_mask: np.ndarray):
    """Run watershed on a single ROI."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL)
    cleaned = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_OPEN_ITERS)
    if cleaned.max() == 0:
        return np.zeros_like(cleaned), None

    seed_mask, n_seeds, dist = find_seed_components(cleaned, k_factor=SEED_K_FACTOR)

    if n_seeds <= 1:
        return cleaned, None

    ret, markers = cv2.connectedComponents(seed_mask)
    markers = markers.astype(np.int32)
    markers = markers + 1   # background 1, seeds >=2

    unknown = cv2.subtract(cleaned, (seed_mask > 0).astype(np.uint8) * 255)
    markers[unknown == 255] = 0

    roi_for_watershed = (roi_img_norm * 255).astype(np.uint8)
    roi_color = cv2.cvtColor(roi_for_watershed, cv2.COLOR_GRAY2BGR)

    try:
        cv2.watershed(roi_color, markers)
    except Exception:
        return cleaned, None

    separated_mask = (markers > 1).astype(np.uint8) * 255
    return separated_mask, markers


def filter_small_components(binary_mask_uint8: np.ndarray, min_area: int = MIN_AREA):
    """Remove connected components smaller than min_area."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask_uint8, connectivity=8)
    out = np.zeros_like(binary_mask_uint8)
    removed = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            out[labels == i] = 255
        else:
            removed += 1
    return out, removed


def process_image_roi_watershed(
    img_path: str,
    merged_peaks_min: int = MERGED_PEAKS_MIN,
    min_area: int = MIN_AREA,
    visualize: bool = False,
    save_debug: bool = False,
):
    """
    img_path: path to grayscale segmentation map (values 0..255 or normalized)
    returns: final_binary_mask (0/255), count_after, count_before, debug_info
    """
    raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(img_path)
    img = raw.astype(np.float32) / 255.0

    k = 0.736
    mean_val = float(np.mean(raw))
    std_val = float(np.std(raw))
    dynamic_thresh_255 = mean_val - k * std_val
    dynamic_thresh_norm = dynamic_thresh_255 / 255.0

    init_mask = simple_threshold_mask(img, dynamic_thresh_norm)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(init_mask, connectivity=8)
    count_before = n_labels - 1

    final_mask = np.zeros_like(init_mask)
    debug_rois = []

    for lab in range(1, n_labels):
        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        w = int(stats[lab, cv2.CC_STAT_WIDTH])
        h = int(stats[lab, cv2.CC_STAT_HEIGHT])

        pad = 2
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(init_mask.shape[1], x + w + pad)
        y1 = min(init_mask.shape[0], y + h + pad)

        roi_mask = init_mask[y0:y1, x0:x1].copy()
        roi_img = img[y0:y1, x0:x1].copy()
        main_blob_mask = (labels[y0:y1, x0:x1] == lab).astype(np.uint8) * 255
        roi_mask = cv2.bitwise_and(roi_mask, main_blob_mask)

        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(roi_mask, contours, -1, (255), thickness=cv2.FILLED)

        seed_mask, n_seeds, dist = find_seed_components(roi_mask, k_factor=SEED_K_FACTOR)
        run_watershed = (n_seeds >= merged_peaks_min)

        if run_watershed:
            separated_mask, markers = local_watershed_on_roi(roi_img, roi_mask)
            if separated_mask is None:
                final_mask[y0:y1, x0:x1] = cv2.bitwise_or(final_mask[y0:y1, x0:x1], roi_mask)
            else:
                separated_mask_filtered, removed_small = filter_small_components(separated_mask, min_area)
                final_mask[y0:y1, x0:x1] = cv2.bitwise_or(final_mask[y0:y1, x0:x1], separated_mask_filtered)

                if visualize or save_debug:
                    debug_rois.append({
                        "bbox": (x0, y0, x1, y1),
                        "roi_img": roi_img,
                        "roi_mask": roi_mask,
                        "dist": dist,
                        "seed_mask": seed_mask,
                        "separated_mask": separated_mask,
                        "separated_filtered": separated_mask_filtered,
                        "markers": markers,
                    })
        else:
            final_mask[y0:y1, x0:x1] = cv2.bitwise_or(final_mask[y0:y1, x0:x1], roi_mask)
            if visualize or save_debug:
                debug_rois.append({
                    "bbox": (x0, y0, x1, y1),
                    "roi_img": roi_img,
                    "roi_mask": roi_mask,
                    "dist": dist,
                    "seed_mask": seed_mask,
                    "separated_mask": None,
                    "separated_filtered": None,
                    "markers": None,
                })

    final_mask_clean, removed_total = filter_small_components(final_mask, min_area)
    n_final, labels_final = cv2.connectedComponents(final_mask_clean, connectivity=8)
    count_after = n_final - 1

    if visualize:
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 3, 1)
        plt.title("Input (normalized)")
        plt.imshow(img, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title(f"Initial Mask count={count_before}")
        plt.imshow(init_mask, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title(f"Final Mask count={count_after}")
        plt.imshow(final_mask_clean, cmap="gray")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        for idx, r in enumerate(debug_rois):
            x0, y0, x1, y1 = r["bbox"]
            plt.figure(figsize=(12, 6))

            plt.subplot(2, 4, 1)
            plt.title("ROI img")
            plt.imshow(r["roi_img"], cmap="gray")
            plt.axis("off")

            plt.subplot(2, 4, 2)
            plt.title("ROI initial mask")
            plt.imshow(r["roi_mask"], cmap="gray")
            plt.axis("off")

            plt.subplot(2, 4, 3)
            plt.title("Distance transform")
            plt.imshow(r["dist"], cmap="magma")
            plt.axis("off")

            plt.subplot(2, 4, 4)
            plt.title("Seed mask")
            plt.imshow(r["seed_mask"], cmap="gray")
            plt.axis("off")

            plt.subplot(2, 4, 5)
            plt.title("Separated (raw)")
            if r["separated_mask"] is None:
                plt.text(0.2, 0.5, "No watershed run", fontsize=12)
                plt.axis("off")
            else:
                plt.imshow(r["separated_mask"], cmap="gray")
                plt.axis("off")

            plt.subplot(2, 4, 6)
            plt.title("Separated (filtered)")
            if r["separated_filtered"] is None:
                plt.text(0.2, 0.5, "No watershed run", fontsize=12)
                plt.axis("off")
            else:
                plt.imshow(r["separated_filtered"], cmap="gray")
                plt.axis("off")

            plt.subplot(2, 4, 7)
            plt.title("Markers (labels)")
            if r["markers"] is None:
                plt.text(0.2, 0.5, "No markers", fontsize=12)
                plt.axis("off")
            else:
                markers_vis = r["markers"].copy()
                markers_vis[markers_vis < 0] = 0
                plt.imshow(markers_vis, cmap="nipy_spectral")
                plt.axis("off")

            plt.subplot(2, 4, 8)
            plt.title("Overview in global img")
            context = np.zeros_like(final_mask_clean)
            context[y0:y1, x0:x1] = (
                r["separated_filtered"] if r["separated_filtered"] is not None else r["roi_mask"]
            )
            plt.imshow(context, cmap="gray")
            plt.axis("off")

            plt.suptitle(f"ROI #{idx+1} bbox={r['bbox']}")
            plt.tight_layout()
            plt.show()

    if save_debug:
        ensure_dir(DEBUG_SAVE_DIR)
        base = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, f"{base}_init_mask.png"), init_mask)
        cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, f"{base}_final_mask.png"), final_mask_clean)
        for idx, r in enumerate(debug_rois):
            x0, y0, x1, y1 = r["bbox"]
            cv2.imwrite(
                os.path.join(DEBUG_SAVE_DIR, f"{base}_roi{idx}_img.png"),
                (r["roi_img"] * 255).astype(np.uint8),
            )
            cv2.imwrite(
                os.path.join(DEBUG_SAVE_DIR, f"{base}_roi{idx}_initmask.png"),
                r["roi_mask"],
            )
            if r["seed_mask"] is not None and r["seed_mask"].max() > 0:
                cv2.imwrite(
                    os.path.join(DEBUG_SAVE_DIR, f"{base}_roi{idx}_seeds.png"),
                    r["seed_mask"],
                )
            if r["separated_mask"] is not None:
                cv2.imwrite(
                    os.path.join(DEBUG_SAVE_DIR, f"{base}_roi{idx}_sep_raw.png"),
                    r["separated_mask"],
                )
            if r["separated_filtered"] is not None:
                cv2.imwrite(
                    os.path.join(DEBUG_SAVE_DIR, f"{base}_roi{idx}_sep_filtered.png"),
                    r["separated_filtered"],
                )

    debug_info = {
        "count_before": count_before,
        "count_after": count_after,
        "removed_small_objects": removed_total,
        "num_rois_examined": len(debug_rois),
    }

    return final_mask_clean, count_after, count_before, debug_info


def plot_froth_regions(final_clean_mask: np.ndarray):
    """Detect and visualize connected froth regions in a binary mask."""
    mask_bool = final_clean_mask > 0
    label_img = label(mask_bool)
    regions = regionprops(label_img)

    print(f"Total detected froth regions: {len(regions)}\n")

    for i, region in enumerate(regions):
        area = region.area
        centroid = region.centroid
        bbox = region.bbox

        print(f"Froth #{i+1}")
        print(f" - Area: {area} pixels")
        print(f" - Centroid: (y={centroid[0]:.2f}, x={centroid[1]:.2f})")
        print(f" - Bounding Box: {bbox}\n")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(label_img, cmap="gray")

    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        rect = patches.Rectangle(
            (minc, minr),
            maxc - minc,
            maxr - minr,
            edgecolor="red",
            facecolor="none",
            linewidth=1.5,
        )
        ax.add_patch(rect)
        cy, cx = region.centroid
        ax.plot(cx, cy, "go", markersize=3)

    ax.set_title(f"Froth Regions: {len(regions)} (with bounding boxes)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return fig, ax
