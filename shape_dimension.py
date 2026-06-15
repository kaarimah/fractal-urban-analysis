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
        raise FileNotFoundError(
            f"Cannot find image: {path}"
        )

    _, bw = cv2.threshold(
        img,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    if invert:
        bw = 255 - bw

    return (bw > 0).astype(np.uint8)


# ----------------------------
# Shape / Capacity Dimension
# ----------------------------
def shape_dimension(binary_img, image_name):

    sizes = [2, 4, 8, 16, 32, 64]

    masses = []

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

        mass = np.sum(resized)

        if mass > 0:
            masses.append(mass)

    sizes = sizes[:len(masses)]

    # ----------------------------
    # Log Transform
    # ----------------------------

    log_size = np.log(
        1 / np.array(sizes)
    )

    log_mass = np.log(
        np.array(masses)
    )

    # ----------------------------
    # Regression
    # ----------------------------

    slope, intercept, r_value, p_value, std_err = linregress(
        log_size,
        log_mass
    )

    shape_dim = slope

    # ----------------------------
    # Dataset
    # ----------------------------

    dataset = pd.DataFrame({
        "Scale_Size": sizes,
        "Mass": masses,
        "Log_1_Size": log_size,
        "Log_Mass": log_mass
    })

    csv_name = (
        f"{image_name}_shape_dimension.csv"
    )

    dataset.to_csv(
        csv_name,
        index=False
    )

    # ----------------------------
    # Results
    # ----------------------------

    print("\nShape Dimension Results")
    print("-----------------------")
    print(
        f"Shape Dimension = "
        f"{shape_dim:.6f}"
    )
    print(
        f"R² = "
        f"{r_value**2:.6f}"
    )
    print(
        f"Dataset saved as: "
        f"{csv_name}"
    )
    print("-----------------------")

    print("\nScale-wise Mass Values")

    for s, m in zip(sizes, masses):

        print(
            f"Scale {s:4d}"
            f" -> Mass = {m}"
        )

    return shape_dim


# ----------------------------
# RUN
# ----------------------------

if __name__ == "__main__":

    image_path = "Banglore_buil.png"

    image_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    img = load_binary_image(
        image_path,
        threshold=127,
        invert=True
    )

    shape_dimension(
        img,
        image_name
    )
