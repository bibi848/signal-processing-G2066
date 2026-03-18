from pathlib import Path
import cv2
from skimage.metrics import structural_similarity as ssim


def compare_images_ssim(image_a_path: str, image_b_path: str) -> float:
    """
    Compare two images with SSIM.
    Returns a score between -1 and 1, where 1 means identical.
    """

    if not Path(image_a_path).exists():
        raise FileNotFoundError(f"File not found: {image_a_path}")
    if not Path(image_b_path).exists():
        raise FileNotFoundError(f"File not found: {image_b_path}")

    # Load as grayscale for basic SSIM comparison
    img1 = cv2.imread(image_a_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_b_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None:
        raise ValueError(f"Could not read image: {image_a_path}")
    if img2 is None:
        raise ValueError(f"Could not read image: {image_b_path}")

    if img1.shape != img2.shape:
        raise ValueError(
            f"Images must have the same dimensions. "
            f"Got {img1.shape} vs {img2.shape}"
        )

    score = ssim(img1, img2)
    return score

print("running")

if __name__ == "__main__":
    IN_DIR  = Path.cwd() /"DATA" / "1D TFM Data" / "Al Pure 10MHz Ex 09032026 Filtered"
    image1 = IN_DIR / "A5_filtered_TFM.png"
    image2 = IN_DIR / "B5_filtered_TFM.png"

    score = compare_images_ssim(image1, image2)
    print(f"SSIM score: {score:.4f}")

    if score > 0.95:
        print("Images are very similar.")
    elif score > 0.80:
        print("Images are somewhat similar.")
    else:
        print("Images are quite different.")