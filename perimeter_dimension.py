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

    return bw

# ----------------------------
# Perimeter measurement
# ----------------------------
def compute_perimeter(binary_img):
    edges = cv2.Canny(binary_img, 100, 200)
    return np.count_nonzero(edges)

# ----------------------------
# Perimeter fractal dimension
# ----------------------------
def perimeter_fractal_dimension(binary_img):
    sizes = [1, 2, 4, 8, 16, 32]
    perimeters = []

    for s in sizes:
        resized = cv2.resize(binary_img,
                              (binary_img.shape[1]//s, binary_img.shape[0]//s),
                              interpolation=cv2.INTER_NEAREST)
        p = compute_perimeter(resized)
        perimeters.append(p)

    sizes = np.array(sizes, dtype=float)
    perimeters = np.array(perimeters, dtype=float)

    x = np.log(1/sizes)
    y = np.log(perimeters)

    slope, intercept = np.polyfit(x, y, 1)
    Dp = slope

    # Plot
    plt.figure()
    plt.plot(x, y, 'o-', label=f"Dp = {Dp:.4f}")
    plt.xlabel("log(1/ε)")
    plt.ylabel("log(Perimeter)")
    plt.title("Perimeter (Boundary) Fractal Dimension")
    plt.grid(True)
    plt.legend()
    plt.show()

    return Dp

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    path = "tokyo_boundary.png"   # change this
    img = load_binary_image(path)
    Dp = perimeter_fractal_dimension(img)
    print("Perimeter Fractal Dimension =", Dp)
