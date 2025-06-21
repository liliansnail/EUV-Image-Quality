mport numpy as np
import matplotlib.pyplot as plt
import cv2
from astropy.io import fits
import os
import pywt
from skimage.util import view_as_windows

from skimage.feature import graycomatrix, graycoprops

import csv
import time
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, ifft2, fftshift
# ==== 图像加载函数 ====
def load_image(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.fits':
        with fits.open(file_path) as hdul:
            data = hdul[0].data
            image = np.nan_to_num(data)
    elif ext == '.npz':
        data = np.load(file_path)
        image = data['arr_0'] if 'arr_0' in data else list(data.values())[0]
    elif ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError("Unsupported file type:", ext)
    image = image.astype(np.float32)
    image -= np.min(image)
    if np.max(image) > 0:
        image /= np.max(image)
    return image


import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_mfgs(image: np.ndarray) -> float:
    """
    中值滤波梯度相似性（MFGS）指标
    """
    median_filtered = cv2.medianBlur(image, 3)

    def gradient_magnitude(img):
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx ** 2 + gy ** 2)

    grad_orig = gradient_magnitude(image.astype(np.float32))
    grad_median = gradient_magnitude(median_filtered.astype(np.float32))

    numerator = np.sum(grad_orig * grad_median)
    denominator = np.sum(grad_orig ** 2 + grad_median ** 2)

    if denominator == 0:
        return 0.0
    return 2 * numerator / denominator


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    峰值信噪比（PSNR）
    """
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0 if img1.dtype == np.uint8 else 1.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    结构相似性（SSIM）
    """
    score, _ = ssim(img1, img2,data_range=1.0,full=True)
    return float(score)


def snr(original: np.ndarray, filtered: np.ndarray) -> float:
    """
    信噪比（SNR）
    """
    signal_power = np.sum(original.astype(np.float32) ** 2)
    noise_power = np.sum((original.astype(np.float32) - filtered.astype(np.float32)) ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)


def evaluate_image_quality(image_path: str) -> dict:
    """
    输入图像路径，输出质量指标
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"图像读取失败: {image_path}")

    median_img = cv2.medianBlur(img, 3)

    return {
        "MFGS": compute_mfgs(img),
        "PSNR": psnr(img, median_img),
        "SSIM": compute_ssim(img, median_img),
        "SNR": snr(img, median_img)
    }




# ==== 多尺度梯度 ====
def adaptive_gradient_map(img):
    sobel = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)**2 + cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)**2
    scharr = cv2.Scharr(img, cv2.CV_32F, 1, 0)**2 + cv2.Scharr(img, cv2.CV_32F, 0, 1)**2
    grad = np.sqrt(sobel + scharr)
    local_mean = cv2.blur(img, (15, 15))
    mask = local_mean > 0.2
    grad[~mask] = 0
    return grad

# ==== 拉普拉斯响应 ====
# ==== MSCN系数计算 ====
def compute_mscn(image, kernel_size=7, sigma=7/6):
    mu = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    sigma_map = np.sqrt(np.abs(cv2.GaussianBlur(image**2, (kernel_size, kernel_size), sigma) - mu**2)) + 1e-8
    return (image - mu) / sigma_map

# ==== 邻域MSCN方向乘积 ====
def mscn_directional_products(mscn):
    shifts = [
        (0, 1),    # 0 deg
        (-1, 1),   # -45 deg
        (-1, 0),   # -90 deg
        (-1, -1)   # -135 deg
    ]
    products = []
    for dy, dx in shifts:
        shifted = np.roll(mscn, shift=(dy, dx), axis=(0, 1))
        products.append(np.mean(mscn * shifted))
    return products  # 返回 4 个方向的乘积均值

# ==== 灰度共生矩阵特征 ====
def glcm_features(img):
    img_u8 = np.uint8(img * 255)
    glcm = graycomatrix(img_u8, [1], [0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    return contrast, homogeneity, energy

# ==== Log-Gabor 滤波器响应 ====
def log_gabor_response(img, frequency=0.25, sigma=0.55):
    rows, cols = img.shape
    u = np.linspace(-0.5, 0.5, cols)
    v = np.linspace(-0.5, 0.5, rows)
    U, V = np.meshgrid(u, v)
    radius = np.sqrt(U**2 + V**2)
    log_gabor = np.exp(- (np.log(radius / frequency)**2) / (2 * np.log(sigma)**2))
    log_gabor[radius == 0] = 0
    img_fft = fftshift(fft2(img))
    filtered = np.real(ifft2(img_fft * log_gabor))
    return np.mean(np.abs(filtered))

def laplacian_response(img, threshold=0.3):
    lap = np.abs(cv2.Laplacian(img, cv2.CV_32F))
    norm_lap = (lap - lap.min()) / (lap.max() - lap.min() + 1e-8)
    mask = img > threshold
    lap_in_active = norm_lap[mask]
    return np.mean(lap_in_active) if len(lap_in_active) > 0 else 0

# ==== 小波熵 ====
def wavelet_entropy(img, wavelet='db1'):
    coeffs = pywt.dwt2(img, wavelet)
    LL, (LH, HL, HH) = coeffs
    def entropy(band):
        energy = np.square(band)
        p = energy / (np.sum(energy) + 1e-8)
        return -np.sum(p * np.log2(p + 1e-8))
    return {
        'LH_entropy': entropy(LH),
        'HL_entropy': entropy(HL),
        'HH_entropy': entropy(HH)
    }

# ==== Tenengrad 局部锐度 ====
def local_tenengrad_sharpness(img, window=32, stride=16):
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    tenengrad = grad_x**2 + grad_y**2
    win_blocks = view_as_windows(tenengrad, (window, window), step=stride)
    bright_blocks = view_as_windows(img, (window, window), step=stride)
    sharpness_list = []
    for i in range(win_blocks.shape[0]):
        for j in range(win_blocks.shape[1]):
            block = bright_blocks[i, j]
            if np.max(block) - np.min(block) < 0.6:
                sharpness_list.append(np.var(win_blocks[i, j]))
    return np.mean(sharpness_list) if sharpness_list else 0

# ==== 区域得分 + 特征提取 ====
def get_top_blocks_with_features(image, block_size=64, top_k=10, alpha=0.5):
    grad_map = adaptive_gradient_map(image)
    h, w = image.shape
    cx, cy = w / 2, h / 2
    R = min(cx, cy)
    r = 0.8 * R
    results = []
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            cx_blk = x + block_size / 2
            cy_blk = y + block_size / 2
            if np.sqrt((cx_blk - cx)**2 + (cy_blk - cy)**2) > r:
                continue
            block_img = image[y:y+block_size, x:x+block_size]
            block_grad = grad_map[y:y+block_size, x:x+block_size]
            mscn = compute_mscn(block_img)
            mscn_prods = mscn_directional_products(mscn)
            contrast, homogeneity, energy = glcm_features(block_img)
            log_gabor = log_gabor_response(block_img)
            median_img = cv2.medianBlur(block_img, 3)

            features = {
                'x': x, 'y': y,
                'score': alpha * np.mean(block_img) + (1 - alpha) * np.mean(block_grad),#亮度梯度值
                'brightness': np.mean(block_img),
                'gradient': np.mean(block_grad),
                'laplacian': laplacian_response(block_img),
                'sharpness': local_tenengrad_sharpness(block_img),
                'mscn_0': mscn_prods[0],
                'mscn_45': mscn_prods[1],
                'mscn_90': mscn_prods[2],
                'mscn_135': mscn_prods[3],
                'glcm_contrast': contrast,
                'glcm_homogeneity': homogeneity,
                'glcm_energy': energy,
                'log_gabor': log_gabor,
                'MFGS': compute_mfgs(block_img),
                'PSNR': psnr(block_img, median_img),
                'SSIM': compute_ssim(block_img, median_img),
                'SNR': snr(block_img, median_img)
            }
            features.update(wavelet_entropy(block_img))
            results.append(features)
    results.sort(key=lambda d: d['score'], reverse=True)
    return results[:top_k]


# ==== 可视化（可选） ====
def draw_blocks(image, blocks, block_size=64):
    img_rgb = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for i, block in enumerate(blocks):
        x, y = block['x'], block['y']
        cv2.rectangle(img_rgb, (x, y), (x+block_size, y+block_size), (0, 255, 0), 2)
        cv2.putText(img_rgb, str(i+1), (x+2, y+14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    plt.imshow(img_rgb)
    plt.title("Top 10 Active Regions (with Features)")
    plt.axis('off')
    plt.show()

# ==== 批量处理文件夹，保存CSV ====


def process_folder(folder_path, out_csv_path='all_images_top_blocks.csv', block_size=64, top_k=5, alpha=0.5, visualize=False):  #top_k=1
    supported_exts = ['.fits', '.npz', '.png', '.jpg', '.jpeg']
    all_results = []
    file_list = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in supported_exts]
    print(f"📂 Found {len(file_list)} supported files in {folder_path}")

    start_time = time.time()  # ← 记录起始时间

    for idx, filename in enumerate(file_list):
        file_path = os.path.join(folder_path, filename)
        try:
            image = load_image(file_path)
            blocks = get_top_blocks_with_features(image, block_size, top_k, alpha)
            for b in blocks:
                b['filename'] = filename
            all_results.extend(blocks)
            print(f"[{idx+1}/{len(file_list)}] ✅ Processed {filename}")
            if visualize:
                draw_blocks(image, blocks, block_size)
        except Exception as e:
            print(f"[{idx+1}/{len(file_list)}] ❌ Failed to process {filename}: {e}")

    elapsed = time.time() - start_time  # ← 总耗时
    print(f"⏱️ Total processing time: {elapsed:.2f} seconds")

    if all_results:
        fieldnames = [
            'filename', 'x', 'y', 'score', 'brightness', 'gradient', 'laplacian', 'sharpness',
            'mscn_0', 'mscn_45', 'mscn_90', 'mscn_135',
            'glcm_contrast', 'glcm_homogeneity', 'glcm_energy',
            'log_gabor',
            'LH_entropy', 'HL_entropy', 'HH_entropy','MFGS','PSNR','SSIM','SNR'
        ]
        with open(out_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_results:
                writer.writerow(row)
        print(f"💾 Saved all results to: {out_csv_path}")
    else:
        print("⚠️ No results to save.")

# ==== 入口示例 ====
if __name__ == '__main__':
    #folder =r"D:\sdo193\img230420"   # 替换成你的文件夹路径
    folder =r"D:\FY03IMG\database\simdegration"
    output_csv = r"D:\FY03IMG\database\simdegration\top5NewfeaturesscoreMFGS.csv"
    #output_csv = r"D:\sdo193\img230420\top10_blocks_NewfeaturesscoreMFGS.csv"
    #folder = r"D:\code\blursun"  # 替换成你的文件夹路径
    #output_csv = r"D:\code\blursun\top_blocks_features_all.csv"
    process_folder(folder, output_csv, visualize=False)
