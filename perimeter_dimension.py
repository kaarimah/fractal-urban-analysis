import cv2
import numpy as np
import pandas as pd
import os
from scipy.stats import linregress

# ----------------------------
# Load binary image
# ----------------------------
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

    return bw


# ----------------------------
# Perimeter measurement
# ----------------------------
def compute_perimeter(binary_img):

    edges = cv2.Canny(binary_img, 100, 200)

    return np.count_nonzero(edges)


# ----------------------------
# Boundary Dimension
# ----------------------------
def boundary_dimension(binary_img, image_name):

    sizes = [1, 2, 4, 8, 16, 32]

    perimeters = []

    for s in sizes:

        new_width = binary_img.shape[1] // s
        new_height = binary_img.shape[0] // s

        if new_width < 2 or new_height < 2:
            break

        resized = cv2.resize(
            binary_img,
            (new_width, new_height),
            interpolation=cv2.INTER_NEAREST
        )

        p = compute_perimeter(resized)

        if p > 0:
            perimeters.append(p)

    sizes = sizes[:len(perimeters)]

    # --------------------------------
    # Log Transform
    # --------------------------------

    log_size = np.log(1 / np.array(sizes))
    log_perimeter = np.log(np.array(perimeters))

    # --------------------------------
    # Linear Regression
    # --------------------------------

    slope, intercept, r_value, p_value, std_err = linregress(
        log_size,
        log_perimeter
    )

    boundary_dim = slope

    # --------------------------------
    # Dataset
    # --------------------------------

    dataset = pd.DataFrame({
        "Scale_Size": sizes,
        "Perimeter": perimeters,
        "Log_1_Size": log_size,
        "Log_Perimeter": log_perimeter
    })

    csv_name = f"{image_name}_boundary_dimension.csv"

    dataset.to_csv(
        csv_name,
        index=False
    )

    # --------------------------------
    # Results
    # --------------------------------

    print("\nBoundary Fractal Dimension Results")
    print("----------------------------------")
    print(f"Boundary Dimension = {boundary_dim:.6f}")
    print(f"R² = {r_value**2:.6f}")
    print(f"Dataset saved as: {csv_name}")
    print("----------------------------------")

    return boundary_dim


# ----------------------------
# RUN
# ----------------------------

if __name__ == "__main__":

    image_path = "banglore_perimeter.png"

    image_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    img = load_binary_image(
        image_path,
        invert=True
    )

    boundary_dimension(
        img,
        image_name
    )
