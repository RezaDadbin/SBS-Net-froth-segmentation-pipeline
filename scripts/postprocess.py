import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
from tqdm import tqdm

from config import Config as C
from sbsnet_froth.utils.postprocess import process_image_roi_watershed


def main():
    input_dir = C.pred_masks_dir
    output_dir = C.postproc_masks_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    image_paths = [
        p for p in sorted(input_dir.iterdir())
        if p.suffix.lower() in exts
    ]

    print(f"Found {len(image_paths)} prediction masks in {input_dir}")

    for img_path in tqdm(image_paths, desc="Postprocess"):
        final_mask, cnt_after, cnt_before, dbg = process_image_roi_watershed(
            str(img_path),
            visualize=False,
            save_debug=False,
        )
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), final_mask)
        print(
            f"{img_path.name}: before={cnt_before}, "
            f"after={cnt_after}, removed_small={dbg['removed_small_objects']}"
        )

    print(f"Saved postprocessed masks to {output_dir}")


if __name__ == "__main__":
    main()
