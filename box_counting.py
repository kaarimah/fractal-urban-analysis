import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Load + preprocess image
# ---------------------------
def load_binary_image(path, threshold=127, invert=True):
    """
    Loads a PNG/JPG image, converts to grayscale and binary.
    invert=True means white=background, black=roads/buildings.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found or unreadable")

    _, bw = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    if invert:
        bw = 255 - bw

    return (bw > 0).astype(np.uint8)

# ---------------------------
# Box Counting
# ---------------------------
def boxcount(Z, k):
    """
    Count the number of non-empty k×k boxes.
    """
    h, w = Z.shape
    h = (h // k) * k
    w = (w // k) * k
    Z = Z[:h, :w]

    Zb = Z.reshape(h//k, k, w//k, k)
    blocks = Zb.sum(axis=(1,3))
    return np.count_nonzero(blocks)

def fractal_dimension_boxcount(Z):
    """
    Estimate box-counting fractal dimension from binary image Z.
    """
    assert Z.ndim == 2

    # Choose box sizes = powers of 2
    p = min(Z.shape)
    n = 2**int(np.floor(np.log2(p)))
    sizes = 2**np.arange(int(np.log2(n)), 1, -1)

    counts = []
    for s in sizes:
        c = boxcount(Z, int(s))
        if c > 0:
            counts.append(c)
        else:
            counts.append(np.nan)

    # Convert to arrays
    sizes = np.array(sizes, dtype=float)
    counts = np.array(counts, dtype=float)

    # Clean invalid values
    mask = np.isfinite(counts) & (counts > 0)
    x = np.log(1/sizes[mask])
    y = np.log(counts[mask])

    # Linear regression
    slope, intercept = np.polyfit(x, y, 1)
    D = slope

    # Plot
    plt.figure(figsize=(6,5))
    plt.scatter(x, y, label=f"D = {D:.4f}")
    plt.plot(x, slope*x + intercept, 'r--')
    plt.xlabel("log(1/ε)")
    plt.ylabel("log(N(ε))")
    plt.title("Box-Counting Fractal Dimension")
    plt.grid(True)
    plt.legend()
    plt.show()

    return D

# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    path = "dubai_highway.png"   # change this
    img = load_binary_image(path, threshold=127, invert=True)
    D = fractal_dimension_boxcount(img)
    print("Box-counting Dimension =", D)
