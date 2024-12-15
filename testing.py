import cv2
import matplotlib.pyplot as plt
from model import model

TEST_IMG_PATH = "images/football.png"
results = model(TEST_IMG_PATH)

# Plot predictions
fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))
ax.axis("off")

plt.show()