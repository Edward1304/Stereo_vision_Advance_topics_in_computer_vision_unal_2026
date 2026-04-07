import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import rawpy


@dataclass
class PipelineResult:
    name: str
    kp1_count: int
    kp2_count: int
    raw_matches: int
    good_matches: int
    inliers: int
    disparity_valid_ratio: float
    disparity_mean: float
    disparity_map: np.ndarray
    rect_left: np.ndarray
    rect_right: np.ndarray
    matches_vis: np.ndarray
    inlier_matches_vis: np.ndarray


def load_gray(path: str) -> np.ndarray:
    if str(path).lower().endswith(".raw"):
        try:
            with rawpy.imread(path) as raw:
                # Usar half_size para reducir resolución (mayor velocidad de proceso)
                rgb = raw.postprocess(half_size=True)
                img = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
                # Escalar la imagen a un tamaño manejable (~1024px)
                h, w = img.shape
                scale = 1024.0 / max(w, h)
                if scale < 1.0:
                    img = cv.resize(img, (0,0), fx=scale, fy=scale)
                return img
        except Exception as e:
            raise RuntimeError(f"Error procesando RAW con rawpy {path}: {e}")

    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No pude abrir la imagen: {path}")
    return img


def create_detector(method: str):
    method = method.upper()
    if method == "SIFT":
        return cv.SIFT_create()

    if method == "SURF":
        if not hasattr(cv, "xfeatures2d"):
            raise RuntimeError(
                "OpenCV no tiene xfeatures2d. Instala opencv-contrib-python."
            )
        if not hasattr(cv.xfeatures2d, "SURF_create"):
            raise RuntimeError(
                "Tu build de OpenCV no expone SURF_create(). "
                "Necesitas una build con xfeatures2d/nonfree."
            )
        # hessianThreshold más alto => menos puntos, más selectivos
        return cv.xfeatures2d.SURF_create(
            hessianThreshold=400,
            nOctaves=4,
            nOctaveLayers=3,
            extended=False,
            upright=False
        )

    raise ValueError("Método no soportado. Usa SIFT o SURF.")


def detect_and_describe(img1: np.ndarray, img2: np.ndarray, method: str):
    detector = create_detector(method)
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        raise RuntimeError(f"{method}: no hay suficientes features para continuar.")

    return kp1, des1, kp2, des2


def knn_ratio_match(des1, des2, ratio=0.75):
    # SIFT/SURF usan descriptores float => NORM_L2
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    return knn, good


def fundamental_ransac(kp1, kp2, matches):
    if len(matches) < 8:
        raise RuntimeError("No hay suficientes matches para estimar F.")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    F, mask = cv.findFundamentalMat(
        pts1, pts2, cv.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.99
    )

    if F is None or mask is None:
        raise RuntimeError("No se pudo estimar la matriz fundamental.")

    mask = mask.ravel().astype(bool)
    inlier_matches = [m for m, keep in zip(matches, mask) if keep]

    pts1_in = pts1[mask]
    pts2_in = pts2[mask]

    if len(pts1_in) < 8:
        raise RuntimeError("Muy pocos inliers después de RANSAC.")

    return F, mask, inlier_matches, pts1_in, pts2_in


def rectify_uncalibrated(img1, img2, pts1, pts2, F):
    h, w = img1.shape[:2]

    ok, H1, H2 = cv.stereoRectifyUncalibrated(
        np.float32(pts1), np.float32(pts2), F, imgSize=(w, h)
    )
    if not ok:
        raise RuntimeError("No se pudo rectificar el par de imágenes.")

    rect1 = cv.warpPerspective(img1, H1, (w, h))
    rect2 = cv.warpPerspective(img2, H2, (w, h))
    return rect1, rect2


def compute_disparity_sgbm(rect1: np.ndarray, rect2: np.ndarray) -> np.ndarray:
    # numDisparities debe ser múltiplo de 16
    min_disp = 0
    num_disp = 16 * 8
    block_size = 5

    matcher = cv.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 1 * block_size * block_size,
        P2=32 * 1 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity = matcher.compute(rect1, rect2).astype(np.float32) / 16.0
    return disparity


def disparity_stats(disparity: np.ndarray):
    valid = disparity > 0
    valid_ratio = float(np.mean(valid))
    mean_disp = float(np.mean(disparity[valid])) if np.any(valid) else 0.0
    return valid_ratio, mean_disp


def normalize_for_display(disparity: np.ndarray) -> np.ndarray:
    disp = disparity.copy()
    valid = disp > 0
    if not np.any(valid):
        return np.zeros_like(disp, dtype=np.uint8)

    dmin = np.min(disp[valid])
    dmax = np.max(disp[valid])
    if dmax - dmin < 1e-6:
        return np.zeros_like(disp, dtype=np.uint8)

    norm = np.zeros_like(disp, dtype=np.float32)
    norm[valid] = (disp[valid] - dmin) / (dmax - dmin)
    return (norm * 255).astype(np.uint8)


def draw_matches(img1, kp1, img2, kp2, matches, max_draw=80):
    draw = matches[:max_draw]
    return cv.drawMatches(
        img1, kp1, img2, kp2, draw, None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )


def run_pipeline(img1_path: str, img2_path: str, method: str) -> PipelineResult:
    img1 = load_gray(img1_path)
    img2 = load_gray(img2_path)

    kp1, des1, kp2, des2 = detect_and_describe(img1, img2, method)
    raw_knn, good = knn_ratio_match(des1, des2, ratio=0.75)

    F, mask, inlier_matches, pts1_in, pts2_in = fundamental_ransac(kp1, kp2, good)
    rect1, rect2 = rectify_uncalibrated(img1, img2, pts1_in, pts2_in, F)
    disparity = compute_disparity_sgbm(rect1, rect2)
    valid_ratio, mean_disp = disparity_stats(disparity)

    matches_vis = draw_matches(img1, kp1, img2, kp2, good)
    inlier_vis = draw_matches(img1, kp1, img2, kp2, inlier_matches)

    return PipelineResult(
        name=method.upper(),
        kp1_count=len(kp1),
        kp2_count=len(kp2),
        raw_matches=len(raw_knn),
        good_matches=len(good),
        inliers=len(inlier_matches),
        disparity_valid_ratio=valid_ratio,
        disparity_mean=mean_disp,
        disparity_map=disparity,
        rect_left=rect1,
        rect_right=rect2,
        matches_vis=matches_vis,
        inlier_matches_vis=inlier_vis
    )


def print_summary(res: PipelineResult):
    print(f"\n=== {res.name} ===")
    print(f"Keypoints img1:           {res.kp1_count}")
    print(f"Keypoints img2:           {res.kp2_count}")
    print(f"Matches KNN:              {res.raw_matches}")
    print(f"Matches tras ratio test:  {res.good_matches}")
    print(f"Inliers RANSAC:           {res.inliers}")
    print(f"Fracción disparidad válida: {res.disparity_valid_ratio:.3f}")
    print(f"Disparidad media válida:    {res.disparity_mean:.3f}")


def compare_results(r1: PipelineResult, r2: PipelineResult):
    print("\n=== Comparación ===")

    def better(a, b, name, higher_is_better=True):
        if abs(a - b) < 1e-9:
            print(f"{name}: empate")
        else:
            if (a > b and higher_is_better) or (a < b and not higher_is_better):
                print(f"{name}: gana {r1.name} ({a:.3f} vs {b:.3f})")
            else:
                print(f"{name}: gana {r2.name} ({b:.3f} vs {a:.3f})")

    better(r1.good_matches, r2.good_matches, "Matches buenos")
    better(r1.inliers, r2.inliers, "Inliers")
    better(r1.disparity_valid_ratio, r2.disparity_valid_ratio, "Densidad de disparidad")
    better(r1.disparity_mean, r2.disparity_mean, "Disparidad media")


def show_result(res: PipelineResult):
    disp_vis = normalize_for_display(res.disparity_map)

    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    plt.title(f"{res.name} - Matches buenos")
    plt.imshow(cv.cvtColor(res.matches_vis, cv.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title(f"{res.name} - Inliers RANSAC")
    plt.imshow(cv.cvtColor(res.inlier_matches_vis, cv.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.title(f"{res.name} - Rectificada izquierda")
    plt.imshow(res.rect_left, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.title(f"{res.name} - Disparidad")
    plt.imshow(disp_vis, cmap="plasma")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import os

    # Adaptado para usar dos imágenes RAW consecutivas de la secuencia
    img1_path = os.path.join("baby", "photo-20251016-214626.raw")
    img2_path = os.path.join("baby", "photo-20251016-214637.raw")

    sift_res = run_pipeline(img1_path, img2_path, "SIFT")
    print_summary(sift_res)

    try:
        surf_res = run_pipeline(img1_path, img2_path, "SURF")
        print_summary(surf_res)
        compare_results(sift_res, surf_res)

        show_result(sift_res)
        show_result(surf_res)

    except Exception as e:
        print("\nSURF no pudo ejecutarse:")
        print(str(e))
        print("\nMostrando solo resultados de SIFT.")
        show_result(sift_res)