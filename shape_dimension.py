import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Load binary image
# ----------------------------
def load_binary_image(path, threshold=127, invert=True):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found")

    _, bw = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    if invert:
        bw = 255 - bw

    return (bw > 0).astype(np.uint8)

# ----------------------------
# Shape (Capacity) Dimension D0
# ----------------------------
def shape_dimension(binary_img):
    sizes = [2, 4, 8, 16, 32, 64]
    masses = []

    for s in sizes:
        resized = cv2.resize(binary_img,
                              (binary_img.shape[1]//s, binary_img.shape[0]//s),
                              interpolation=cv2.INTER_NEAREST)
        mass = np.sum(resized)
        masses.append(mass)

    sizes = np.array(sizes, dtype=float)
    masses = np.array(masses, dtype=float)

    x = np.log(1/sizes)
    y = np.log(masses)

    slope, intercept = np.polyfit(x, y, 1)
    Ds = slope

    # Plot
    plt.figure()
    plt.plot(x, y, 'o-', label=f"Ds = {Ds:.4f}")
    plt.xlabel("log(1/ε)")
    plt.ylabel("log(Mass)")
    plt.title("Shape / Capacity Dimension")
    plt.grid(True)
    plt.legend()
    plt.show()

    return Ds

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    path = "tokyo_boundary.png"   # change this
    img = load_binary_image(path)
    Ds = shape_dimension(img)
    print("Shape (Capacity) Dimension =", Ds)
