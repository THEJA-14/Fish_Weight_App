import cv2
import numpy as np

def extract_measurements_pixels(image_path, debug=False):
    """Extract basic measurements in PIXELS from an image of a single fish on a plain background.

    Returns a dict with Length3 (longest axis), Height (bbox height), Width (bbox width),
    and approximate Length1/Length2 as fractions of Length3.

    NOTE: This returns measurements in PIXELS. Convert to cm by dividing by pixels_per_cm
    (provided by user) or calibrate using a ruler present in the image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    h_img, w_img = img.shape[:2]

    # Resize for faster processing while keeping aspect ratio (but keep scale consistent)
    scale = 1.0
    max_dim = 1200
    if max(h_img, w_img) > max_dim:
        scale = max_dim / max(h_img, w_img)
        img = cv2.resize(img, (int(w_img*scale), int(h_img*scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Use morphological operations + adaptive threshold for robustness
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert if background is dark
    white_ratio = np.sum(thresh==255) / thresh.size
    if white_ratio < 0.5:
        thresh = 255 - thresh

    # Clean small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Assume largest contour is the fish
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 100:  # too small
        return None

    # Minimum area rectangle and its size
    rect = cv2.minAreaRect(c)
    (cx, cy), (w, h), angle = rect
    length_px = max(w, h)
    width_px  = min(w, h)
    bbox_w, bbox_h = w, h

    # Principal axis via PCA on contour points for more robust major axis
    data_pts = c.reshape(-1,2).astype(np.float64)
    data_pts_mean = data_pts.mean(axis=0)
    cov = np.cov((data_pts - data_pts_mean).T)
    eigvals, eigvecs = np.linalg.eig(cov)
    major_vec = eigvecs[:, np.argmax(eigvals)]
    # Project contour points onto major axis to find extreme points
    projections = (data_pts - data_pts_mean) @ major_vec
    min_p = projections.min()
    max_p = projections.max()
    # Length along major axis in pixels
    length_major = max_p - min_p

    # Height: bbox_h (smaller axis) approximated as body height in px
    height_px = bbox_h

    # Estimate length1 and length2 as fractions (placeholders)
    length1_px = length_major * 0.90
    length2_px = length_major * 0.97

    result = {
        "Length3_px": float(length_major),
        "Length1_px": float(length1_px),
        "Length2_px": float(length2_px),
        "Height_px": float(height_px),
        "Width_px": float(width_px),
        "contour_area": float(area),
        "angle_deg": float(angle)
    }
    if debug:
        result["debug_img_shape"] = img.shape
    # If we scaled down the image, convert back to original pixel scale
    if scale != 1.0:
        inv_scale = 1.0/scale
        for k in list(result.keys()):
            if k.endswith("_px"):
                result[k] = result[k] * inv_scale
    return result

if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python extract_measurements.py path/to/image.jpg")
        sys.exit(0)
    img = sys.argv[1]
    res = extract_measurements_pixels(img, debug=True)
    print(json.dumps(res, indent=2))
