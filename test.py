import numpy as np
import cv2 as cv
import argparse

def dog(img, size=(0,0),sigma_scale=1.6, sigma=2.0, tau=1.0):
    img_float = img.astype(np.float64) / 255.0
    img1 = cv.GaussianBlur(img_float,size,sigma)
    img2 = cv.GaussianBlur(img_float,size,sigma*sigma_scale)
    return img1 - tau * img2

def xdog(img,sigma=2.0,sigma_scale=1.6,tau=1.0,threshold=0.015):
    d = dog(img, sigma=sigma, sigma_scale=sigma_scale, tau=tau)
    return np.where(d < -threshold, 1.0, 0.0)

def sobel_edges(img, ksize=3):
    grad_x = cv.Sobel(img, ddepth=cv.CV_64F, dx=1, dy=0, ksize=ksize)
    grad_y = cv.Sobel(img, ddepth=cv.CV_64F, dx=0, dy=1, ksize=ksize)
    magnitude = cv.magnitude(grad_x, grad_y)
    has_edge = magnitude > 0.0
    theta = np.arctan2(grad_y, grad_x) / np.pi
    abs_theta = np.abs(theta)

    direction_id = np.zeros_like(img, dtype=np.int32)
    is_vertical = (abs_theta < 0.125) | (abs_theta > 0.875)
    is_horizontal = (abs_theta > 0.375) & (abs_theta < 0.625)
    is_diag_slash = (~is_vertical) & (~is_horizontal) & ((grad_x * grad_y) > 0)
    is_diag_backslash = (~is_vertical) & (~is_horizontal) & ((grad_x * grad_y) <= 0)

    direction_id[has_edge & is_vertical] = 1
    direction_id[has_edge & is_horizontal] = 2
    direction_id[has_edge & is_diag_slash] = 3
    direction_id[has_edge & is_diag_backslash] = 4
    return direction_id

def edge_detector(img,edge_threshold=12):
    h,w = img.shape
    h8, w8 = (h//8) * 8, (w//8) * 8
    img = img[:h8, :w8]
    xdog_mask = xdog(img)
    direction_id = sobel_edges(xdog_mask)
    ny, nx = h8//8, w8//8
    blocks = direction_id.reshape(ny, 8, nx, 8).transpose(0, 2, 1, 3).reshape(ny, nx, 64)
    votes = np.zeros((ny, nx, 4), dtype=np.int32)
    for d in range(1, 5):
        votes[:, :, d - 1] = np.sum(blocks == d, axis=-1)
    max_votes = np.max(votes, axis=-1)
    dominant_dir = np.argmax(votes, axis=-1) + 1
    final_id = np.where(max_votes > edge_threshold, dominant_dir, 0)
    return final_id * 0.25

def grayscale_image(inp, scale=0.125):
    image = cv.imread(inp)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h, w = image.shape
    image = image[:(h//8) * 8, :(w//8) * 8]
    if scale != 1.0:
        image = cv.resize(image, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    return image

def apply_low_contrast_tint(gray_out, bg_bgr=(34, 30, 28), fg_bgr=(195, 205, 210)):
    alpha = (gray_out.astype(np.float32) / 255.0)[:, :, None]
    bg = np.array(bg_bgr, dtype=np.float32)
    fg = np.array(fg_bgr, dtype=np.float32)
    colored = bg * (1.0 - alpha) + fg * alpha
    return colored.astype(np.uint8)

def apply_soft_original_colors(inp, gray_out, bg_bgr=(30, 28, 28), saturation=0.55, max_brightness=0.80):
    h_out, w_out = gray_out.shape
    ny, nx = h_out // 8, w_out // 8
    bgr = cv.imread(inp)
    bgr = bgr[:h_out, :w_out]
    small_bgr = cv.resize(bgr, (nx, ny), interpolation=cv.INTER_AREA)
    hsv = cv.cvtColor(small_bgr, cv.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= saturation
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * max_brightness, 60, 210)
    soft_small = cv.cvtColor(hsv.astype(np.uint8), cv.COLOR_HSV2BGR)
    fg_full = cv.resize(soft_small, (w_out, h_out), interpolation=cv.INTER_NEAREST).astype(np.float32)
    bg = np.array(bg_bgr, dtype=np.float32)
    alpha = (gray_out.astype(np.float32) / 255.0)[:, :, None]
    colored = bg * (1.0 - alpha) + fg_full * alpha
    return colored.astype(np.uint8)

def graytoascii(inp):
    ascii_fill = cv.imread("textures/fillASCII.png", cv.IMREAD_GRAYSCALE)
    ascii_edges = cv.imread("textures/edgesASCII.png", cv.IMREAD_GRAYSCALE)
    full_gray = grayscale_image(inp, scale = 1.0)
    small_gray = grayscale_image(inp)
    edge_grid = edge_detector(full_gray, edge_threshold=11)
    ny, nx = edge_grid.shape
    out = np.zeros((ny * 8, nx * 8), dtype=np.uint8)
    for y in range(ny):
        for x in range(nx):
            edge_val = edge_grid[y,x]
            y0, y1 = y * 8, (y + 1) * 8
            x0, x1 = x * 8, (x + 1) * 8
            if edge_val > 0.0:
                tile_idx = int(round(edge_val * 4))
                tx0, tx1 = tile_idx * 8, (tile_idx + 1) * 8
                out[y0:y1, x0:x1] = ascii_edges[:, tx0:tx1]
            else:
                intensity = float(small_gray[y,x]) / 255.0
                intensity = np.clip((intensity - 0.08) / 0.92, 0.0, 1.0) ** 1.5
                tile_idx = round(intensity * 9)
                tx0, tx1 = tile_idx * 8, (tile_idx + 1) * 8
                out[y0:y1, x0:x1] = ascii_fill[:, tx0:tx1]
    #out = apply_low_contrast_tint(out, bg_bgr=(24,28,36), fg_bgr=(216,206,196))
    return out
    out = apply_soft_original_colors(inp, out)
    cv.imshow('image',out)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "--input", "-i",
        required=True,)
    group.add_argument(
        "--grayscale",
        action="store_true"
    )
    group.add_argument(
        "--lowcontrast",
        action="store_true"
    )
    group.add_argument(
        "--imagecolors",
        action="store_true"
    )
    parser.add_argument(
        "--output", "-o",
        nargs="?",
        const="output.png",
        default=None
    )
    args = parser.parse_args()
    ascii = graytoascii(args.input)
    if args.lowcontrast:
        ascii = apply_low_contrast_tint(ascii)
    elif args.imagecolors:
        ascii = apply_soft_original_colors(args.input, ascii)
    if args.output is not None:
        cv.imwrite(args.output, ascii)
    cv.imshow('ascii', ascii)
    cv.waitKey(0)
    cv.destroyAllWindows()