import cv2
import numpy as np
import pandas as pd
import os

# ---------------------------
# Load image and binarize
# ---------------------------
def load_binary_image(path, threshold=127, invert=True):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Cannot find image: {path}")

    _, bw = cv2.threshold(
        img,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    if invert:
        bw = 255 - bw

    return (bw > 0).astype(np.uint8)


# ---------------------------
# Gliding Box Lacunarity
# ---------------------------
def lacunarity_gliding_box(img, box_sizes):

    lac = []

    for k in box_sizes:

        s = cv2.boxFilter(
            img.astype(np.float32),
            ddepth=-1,
            ksize=(k, k),
            normalize=False
        )

        crop = k // 2

        if crop > 0:
            s = s[crop:-crop, crop:-crop]

        m1 = np.mean(s)
        m2 = np.mean(s * s)

        if m1 == 0:
            lac.append(np.nan)
        else:
            lac.append(m2 / (m1 * m1))

    return np.array(box_sizes), np.array(lac)


# ---------------------------
# Main Analysis
# ---------------------------
def mean_lacunarity_analysis(img, image_name):

    h, w = img.shape

    box_sizes = [
        int(h/x)
        for x in [64, 48, 32, 24, 16, 12, 8]
    ]

    box_sizes = [b for b in box_sizes if b > 1]

    sizes, lac = lacunarity_gliding_box(
        img,
        box_sizes
    )

    mean_lacunarity = np.nanmean(lac)

    # -----------------------
    # Dataset
    # -----------------------

    dataset = pd.DataFrame({
        "Box_Size": sizes,
        "Lacunarity": lac,
        "Log_Box_Size": np.log(sizes),
        "Log_Lacunarity": np.log(lac)
    })

    csv_name = (
        f"{image_name}_mean_lacunarity.csv"
    )

    dataset.to_csv(
        csv_name,
        index=False
    )

    # -----------------------
    # Results
    # -----------------------

    print("\nMean Lacunarity Results")
    print("------------------------")
    print(
        f"Mean Lacunarity = "
        f"{mean_lacunarity:.6f}"
    )
    print(
        f"Dataset saved as: "
        f"{csv_name}"
    )
    print("------------------------")

    # Print individual values

    print("\nScale-wise Lacunarity")

    for s, l in zip(sizes, lac):

        print(
            f"Box Size {s:4d}"
            f" -> Lacunarity = {l:.6f}"
        )

    return mean_lacunarity


# ---------------------------
# RUN
# ---------------------------

if __name__ == "__main__":

    image_path = "tokyo_greenspaces.png"

    image_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    img = load_binary_image(
        image_path,
        threshold=127,
        invert=True
    )

    mean_lacunarity_analysis(
        img,
        image_name
    )
