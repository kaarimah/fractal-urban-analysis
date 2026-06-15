import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import linregress

# -----------------------------------
# Load PNG Image
# -----------------------------------

image_path = "highway_istanbul.png"

img = Image.open(image_path).convert("L")
img = np.array(img)

# Convert to binary
# Black urban pixels = 1
# White background = 0

binary = img < 128

# -----------------------------------
# Box Counting Function
# -----------------------------------

def boxcount(Z, k):

    rows = Z.shape[0] // k
    cols = Z.shape[1] // k

    count = 0

    for i in range(rows):
        for j in range(cols):

            box = Z[
                i*k:(i+1)*k,
                j*k:(j+1)*k
            ]

            if np.any(box):
                count += 1

    return count

# -----------------------------------
# Generate Box Sizes
# -----------------------------------

p = min(binary.shape)

sizes = []

k = 2

while k <= p // 2:
    sizes.append(k)
    k *= 2

# -----------------------------------
# Count Occupied Boxes
# -----------------------------------

counts = []

for size in sizes:
    counts.append(boxcount(binary, size))

# -----------------------------------
# Create Dataset
# -----------------------------------

log_size = np.log(1/np.array(sizes))
log_count = np.log(np.array(counts))

dataset = pd.DataFrame({
    "Box_Size": sizes,
    "Box_Count": counts,
    "Log_1_Size": log_size,
    "Log_Count": log_count
})

dataset.to_csv(
    "istanbul_box_counting_dataset.csv",
    index=False
)

# -----------------------------------
# Fractal Dimension
# -----------------------------------

slope, intercept, r_value, p_value, std_err = linregress(
    log_size,
    log_count
)

print("\nBox Counting Dimension =", slope)
print("R² =", r_value**2)

print("\nDataset saved as:")
print("istanbul_box_counting_dataset.csv")
