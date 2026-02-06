import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Load image and binarize
# ---------------------------
def load_binary_image(path, threshold=127, invert=True):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found")

    _, bw = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    if invert:
        bw = 255 - bw

    return (bw > 0).astype(np.uint8)

# ---------------------------
# Lacunarity using gliding box
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
# RUN
# ---------------------------
if __name__ == "__main__":
    path = "tokyo_boundary.png"   # change this
    img = load_binary_image(path, threshold=127, invert=True)

    # Box sizes chosen relative to image size
    h, w = img.shape
    box_sizes = [int(h/x) for x in [64, 48, 32, 24, 16, 12, 8]]

    sizes, lac = lacunarity_gliding_box(img, box_sizes)

    # ---------------------------
    # SINGLE REPRESENTATIVE VALUE
    # ---------------------------
    mean_lacunarity = np.nanmean(lac)

    print("Mean Lacunarity =", mean_lacunarity)

    # ---------------------------
    # Plot (still important)
    # ---------------------------
    plt.figure(figsize=(6,5))
    plt.plot(sizes, lac, 'o-', label=f"Mean Λ = {mean_lacunarity:.4f}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Box size (log)")
    plt.ylabel("Lacunarity Λ(r) (log)")
    plt.title("Lacunarity Analysis")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Print individual values
    for s, l in zip(sizes, lac):
        print(f"Box size {s:4d} → Lacunarity = {l:.4f}")
