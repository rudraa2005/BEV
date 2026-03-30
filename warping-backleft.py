import cv2
import numpy as np

# -------------------------------
# 1. Load image
# -------------------------------
image = cv2.imread("image90-bl.jpg")
if image is None:
    raise ValueError("Image not found")

vis = image.copy()  # for drawing

# -------------------------------
# 2. Define source points
# IMPORTANT: order doesn't matter now (we'll fix it)
# -------------------------------
src = np.float32([[25, 879], [1577, 869], [1588, 665], [9, 595]])

# -------------------------------
# 3. Function to order points properly
# (bottom-left, bottom-right, top-left, top-right)
# -------------------------------
def order_points(pts):
    pts = pts.reshape(4, 2)

    # sort by y (top vs bottom)
    sorted_y = pts[np.argsort(pts[:, 1])]

    top = sorted_y[:2]
    bottom = sorted_y[2:]

    # sort left-right within each
    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]

    tl, tr = top
    bl, br = bottom

    return np.array([bl, br, tl, tr], dtype=np.float32)

src = order_points(src)

# -------------------------------
# 4. Destination (BEV plane)
# -------------------------------
width = 400
height = 800

dst = np.array([
    [0, 0],
    [width, 0],
    [0, height],
    [width, height]
], dtype=np.float32)

# -------------------------------
# 5. Compute Homography
# -------------------------------
H = cv2.getPerspectiveTransform(src, dst)

# -------------------------------
# 6. Warp
# -------------------------------
bev = cv2.warpPerspective(image, H, (width, height))

# -------------------------------
# 7. Visualization (debug)
# -------------------------------

# Draw points + index
for i, p in enumerate(src):
    p = tuple(p.astype(int))
    cv2.circle(vis, p, 6, (0, 0, 255), -1)
    cv2.putText(vis, str(i), p,
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

# Draw edges (no crossing should happen)
cv2.line(vis, tuple(src[0].astype(int)), tuple(src[1].astype(int)), (0,255,0), 2)  # bottom
cv2.line(vis, tuple(src[2].astype(int)), tuple(src[3].astype(int)), (0,255,0), 2)  # top
cv2.line(vis, tuple(src[0].astype(int)), tuple(src[2].astype(int)), (255,0,0), 2)  # left
cv2.line(vis, tuple(src[1].astype(int)), tuple(src[3].astype(int)), (255,0,0), 2)  # right

# -------------------------------
# 8. Show results
# -------------------------------
cv2.imshow("Original + Ordered Points", vis)
cv2.imshow("BEV (Top View)", bev)

cv2.waitKey(0)
cv2.destroyAllWindows()
np.save("H_backleft.npy", H)