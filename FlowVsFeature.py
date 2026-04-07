import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import rawpy


@dataclass
class PipelineResult:
    name: str
    total_points: int
    valid_matches: int
    inliers: int
    disparity_valid_ratio: float
    disparity_mean: float
    disparity_map: np.ndarray
    rect_left: np.ndarray
    rect_right: np.ndarray
    vis_matches: np.ndarray


def load_gray(path: str) -> np.ndarray:
    if path.lower().endswith(".raw"):
        try:
            with rawpy.imread(path) as raw:
                # Usar half_size para acelerar el proceso con imágenes tan grandes
                rgb = raw.postprocess(half_size=True)
                img = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
                # Escalar la imagen a un tamaño manejable (~1024px) para algoritmos pesados de CV
                h, w = img.shape
                scale = 1024.0 / max(w, h)
                if scale < 1.0:
                    img = cv.resize(img, (0,0), fx=scale, fy=scale)
                return img
        except Exception as e:
            raise RuntimeError(f"Error procesando RAW {path}: {e}")

    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No pude abrir la imagen: {path}")
    return img


def draw_epi_matches(img1, pts1, img2, pts2, max_draw=100):
    """
    Dibuja matches como líneas entre dos imágenes concatenadas.
    """
    img1_c = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2_c = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    h1, w1 = img1_c.shape[:2]
    h2, w2 = img2_c.shape[:2]

    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1_c
    canvas[:h2, w1:w1 + w2] = img2_c

    n = min(len(pts1), max_draw)
    idx = np.linspace(0, len(pts1) - 1, n, dtype=int) if len(pts1) > 0 else []

    for i in idx:
        p1 = tuple(np.round(pts1[i]).astype(int))
        p2 = tuple(np.round(pts2[i]).astype(int))
        p2_shift = (p2[0] + w1, p2[1])

        color = tuple(int(x) for x in np.random.randint(0, 255, size=3))
        cv.circle(canvas, p1, 3, color, -1)
        cv.circle(canvas, p2_shift, 3, color, -1)
        cv.line(canvas, p1, p2_shift, color, 1)

    return canvas


def fundamental_ransac(pts1: np.ndarray, pts2: np.ndarray):
    if len(pts1) < 8:
        raise RuntimeError("No hay suficientes correspondencias para estimar F.")

    F, mask = cv.findFundamentalMat(
        pts1, pts2, cv.FM_RANSAC, 1.0, 0.99
    )
    if F is None or mask is None:
        raise RuntimeError("No se pudo estimar la matriz fundamental.")

    mask = mask.ravel().astype(bool)
    pts1_in = pts1[mask]
    pts2_in = pts2[mask]

    if len(pts1_in) < 8:
        raise RuntimeError("Muy pocos inliers después de RANSAC.")

    return F, pts1_in, pts2_in, mask


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


def compute_disparity(rect1, rect2):
    min_disp = 0
    num_disp = 16 * 8
    block_size = 5

    matcher = cv.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity = matcher.compute(rect1, rect2).astype(np.float32) / 16.0
    return disparity


def disparity_stats(disparity):
    valid = disparity > 0
    valid_ratio = float(np.mean(valid))
    mean_disp = float(np.mean(disparity[valid])) if np.any(valid) else 0.0
    return valid_ratio, mean_disp


def normalize_disparity_for_display(disparity):
    disp = disparity.copy()
    valid = disp > 0
    if not np.any(valid):
        return np.zeros_like(disp, dtype=np.uint8)

    dmin = np.min(disp[valid])
    dmax = np.max(disp[valid])

    out = np.zeros_like(disp, dtype=np.float32)
    if dmax > dmin:
        out[valid] = (disp[valid] - dmin) / (dmax - dmin)

    return (255 * out).astype(np.uint8)


# =========================
# PIPELINE 1: OPTICAL FLOW
# =========================

def optical_flow_pipeline(img1_path, img2_path):
    img1 = load_gray(img1_path)
    img2 = load_gray(img2_path)

    # Detectar puntos buenos para trackear en la primera imagen
    pts1 = cv.goodFeaturesToTrack(
        img1,
        maxCorners=3000,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7
    )
    if pts1 is None or len(pts1) < 20:
        raise RuntimeError("No se detectaron suficientes puntos para optical flow.")

    pts1 = pts1.reshape(-1, 2)

    # Lucas-Kanade piramidal
    pts2, status, err = cv.calcOpticalFlowPyrLK(
        img1, img2,
        pts1.astype(np.float32).reshape(-1, 1, 2),
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    if pts2 is None or status is None:
        raise RuntimeError("Falló el cálculo de optical flow.")

    pts2 = pts2.reshape(-1, 2)
    status = status.reshape(-1).astype(bool)

    pts1_valid = pts1[status]
    pts2_valid = pts2[status]

    if len(pts1_valid) < 8:
        raise RuntimeError("Muy pocas correspondencias válidas tras optical flow.")

    F, pts1_in, pts2_in, mask = fundamental_ransac(pts1_valid, pts2_valid)
    rect1, rect2 = rectify_uncalibrated(img1, img2, pts1_in, pts2_in, F)
    disparity = compute_disparity(rect1, rect2)
    valid_ratio, mean_disp = disparity_stats(disparity)

    vis = draw_epi_matches(img1, pts1_in, img2, pts2_in)

    return PipelineResult(
        name="Optical Flow",
        total_points=len(pts1),
        valid_matches=len(pts1_valid),
        inliers=len(pts1_in),
        disparity_valid_ratio=valid_ratio,
        disparity_mean=mean_disp,
        disparity_map=disparity,
        rect_left=rect1,
        rect_right=rect2,
        vis_matches=vis
    )


# =========================
# PIPELINE 2: FEATURES
# =========================

def create_feature_detector(method="SIFT"):
    method = method.upper()

    if method == "SIFT":
        return cv.SIFT_create(), cv.NORM_L2

    if method == "SURF":
        if not hasattr(cv, "xfeatures2d") or not hasattr(cv.xfeatures2d, "SURF_create"):
            raise RuntimeError(
                "SURF no está disponible en esta instalación de OpenCV."
            )
        detector = cv.xfeatures2d.SURF_create(
            hessianThreshold=400,
            nOctaves=4,
            nOctaveLayers=3,
            extended=False,
            upright=False
        )
        return detector, cv.NORM_L2

    raise ValueError("Método de features no soportado. Usa SIFT o SURF.")


def feature_pipeline(img1_path, img2_path, method="SIFT"):
    img1 = load_gray(img1_path)
    img2 = load_gray(img2_path)

    detector, norm_type = create_feature_detector(method)
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        raise RuntimeError("No hay suficientes keypoints/descriptores.")

    matcher = cv.BFMatcher(norm_type, crossCheck=False)
    knn = matcher.knnMatch(des1, des2, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 8:
        raise RuntimeError("Muy pocos matches tras ratio test.")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    F, pts1_in, pts2_in, mask = fundamental_ransac(pts1, pts2)
    rect1, rect2 = rectify_uncalibrated(img1, img2, pts1_in, pts2_in, F)
    disparity = compute_disparity(rect1, rect2)
    valid_ratio, mean_disp = disparity_stats(disparity)

    vis = draw_epi_matches(img1, pts1_in, img2, pts2_in)

    return PipelineResult(
        name=f"Features ({method.upper()})",
        total_points=len(kp1),
        valid_matches=len(good),
        inliers=len(pts1_in),
        disparity_valid_ratio=valid_ratio,
        disparity_mean=mean_disp,
        disparity_map=disparity,
        rect_left=rect1,
        rect_right=rect2,
        vis_matches=vis
    )


def print_summary(res: PipelineResult):
    print(f"\n=== {res.name} ===")
    print(f"Puntos / keypoints iniciales: {res.total_points}")
    print(f"Correspondencias válidas:     {res.valid_matches}")
    print(f"Inliers geométricos:          {res.inliers}")
    print(f"Fracción disparidad válida:   {res.disparity_valid_ratio:.3f}")
    print(f"Disparidad media:             {res.disparity_mean:.3f}")


def compare_results(a: PipelineResult, b: PipelineResult):
    print("\n=== Comparación global ===")

    def show_metric(label, va, vb):
        winner = a.name if va > vb else b.name if vb > va else "Empate"
        print(f"{label}: {va:.3f} vs {vb:.3f} -> {winner}")

    show_metric("Correspondencias válidas", a.valid_matches, b.valid_matches)
    show_metric("Inliers", a.inliers, b.inliers)
    show_metric("Densidad disparidad", a.disparity_valid_ratio, b.disparity_valid_ratio)
    show_metric("Disparidad media", a.disparity_mean, b.disparity_mean)


def show_results(results):
    n = len(results)
    plt.figure(figsize=(15, 5 * n))

    for i, res in enumerate(results):
        disp_vis = normalize_disparity_for_display(res.disparity_map)

        plt.subplot(n, 3, 3 * i + 1)
        plt.title(f"{res.name} - Matches/Inliers")
        plt.imshow(cv.cvtColor(res.vis_matches, cv.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(n, 3, 3 * i + 2)
        plt.title(f"{res.name} - Rectificada izquierda")
        plt.imshow(res.rect_left, cmap="gray")
        plt.axis("off")

        plt.subplot(n, 3, 3 * i + 3)
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

    # Pipeline Optical Flow
    flow_res = optical_flow_pipeline(img1_path, img2_path)
    print_summary(flow_res)

    # Pipeline Features
    feat_res = feature_pipeline(img1_path, img2_path, method="SIFT")
    print_summary(feat_res)

    compare_results(flow_res, feat_res)
    show_results([flow_res, feat_res])