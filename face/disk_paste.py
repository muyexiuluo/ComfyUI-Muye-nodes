# 由 大龙虾 于 2026-09-18 创建，用于 黑框问题修复：脸从 ComfyUI 预览落盘文件读，原图/mask/box 走内存
import os
import time
import glob
import cv2
import numpy as np
import torch

# ---- 可移植路径：从本节点文件位置推导 ComfyUI 根 ----
# 本文件: <COMFY_ROOT>/custom_nodes/<plugin>/face/disk_paste.py
FACE_DIR = os.path.dirname(os.path.abspath(__file__))
PLUGIN_DIR = os.path.dirname(FACE_DIR)
CUSTOM_NODES_DIR = os.path.dirname(PLUGIN_DIR)
COMFY_ROOT = os.path.dirname(CUSTOM_NODES_DIR)
# 兜底：若目录结构不符（被人挪过位置），向上找含 main.py 的目录
_d = os.path.dirname(COMFY_ROOT)
for _ in range(4):
    if os.path.isfile(os.path.join(_d, 'main.py')):
        COMFY_ROOT = _d
        break
    _d = os.path.dirname(_d)

SCAN_DIRS = [os.path.join(COMFY_ROOT, 'temp'), os.path.join(COMFY_ROOT, 'output')]
FILE_WINDOW_SEC = 60.0   # 只收 60s 内的预览文件（trigger 保证本次 run 的 <2s 前落盘）
EVIDENCE_DIR = os.path.join(COMFY_ROOT, 'temp', 'muye_face_paste_disk_evidence')


def _img_to_uint8(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    a = np.asarray(img)
    while a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    a = a.astype(np.float32)
    if a.max() <= 1.0:
        a = a * 255.0
    return np.clip(a, 0, 255).round().astype(np.uint8)


def _mask01(m):
    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().numpy()
    a = np.asarray(m)
    a = np.squeeze(a)
    if a.ndim == 3:
        a = a[0] if a.shape[0] == 1 else a[..., 0]
    a = a.astype(np.float32)
    if a.max() > 1.0:
        a = a / 255.0
    return a


def _pick_preview_png(w, h):
    """扫 temp/output，找 (h,w) 尺寸匹配、60s 内最新的预览 PNG（= 954 的好脸落盘文件）"""
    now = time.time()
    best = (0.0, None)
    for d in SCAN_DIRS:
        if not os.path.isdir(d):
            continue
        for p in glob.glob(os.path.join(d, '*.png')):
            try:
                st = os.stat(p)
                if now - st.st_mtime > FILE_WINDOW_SEC:
                    continue
                if st.st_mtime <= best[0]:
                    continue
                img = cv2.imread(p, cv2.IMREAD_COLOR)
                if img is not None and img.shape[0] == h and img.shape[1] == w:
                    best = (st.mtime if hasattr(st, 'mtime') else st.st_mtime, p)
            except Exception:
                continue
    return best[1]


class 面部粘贴磁盘读取:
    """终版: 脸从磁盘预览文件读（40/40 验证好脸），原图/mask/box 走内存（40/40 验证可靠）。
    trigger 必须接 954 预览输出，强制其先执行落盘。"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "原图": ("IMAGE",),
                "裁剪图像": ("IMAGE",),     # 仅执行顺序锚点，内容不读取
                "裁剪数据": ("FACE_CROP_DATA",),
                "裁剪遮罩": ("MASK",),
            },
            "optional": {
                "trigger": ("IMAGE",),      # 接 954 预览输出
            },
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("输出图像",)
    FUNCTION = "run"
    CATEGORY = "Muye/面部"

    def run(self, 原图, 裁剪图像, 裁剪数据, 裁剪遮罩, trigger=None):
        if not (isinstance(裁剪数据, dict) and 裁剪数据.get("box")):
            print('[面部粘贴（磁盘读取）] 裁剪数据无效，原样返回', flush=True)
            return (原图,)
        x, y, w, h = 裁剪数据["box"]
        angle = 裁剪数据.get("angle", 0)
        center = 裁剪数据.get("center", None)
        rotated = 裁剪数据.get("rotated", False)

        # ---- 取 crop 的真实尺寸（从内存 tensor 的 shape 读，shape 可靠，内容不可靠）----
        try:
            a0 = 裁剪图像.detach().cpu().numpy()
            while a0.ndim == 4 and a0.shape[0] == 1:
                a0 = a0[0]
            ch, cw = a0.shape[0], a0.shape[1]
        except Exception:
            ch, cw = h, w

        # ---- 脸：从本次 run 的 954 预览落盘文件读（trigger 保证已落盘）----
        src_path = _pick_preview_png(cw, ch)
        if src_path is None:
            print('[面部粘贴（磁盘读取）] 未找到 %dx%d 预览文件（60s 窗口），原样返回' % (cw, ch), flush=True)
            return (原图,)
        crop_u8 = cv2.cvtColor(cv2.imread(src_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if crop_u8 is None:
            print('[面部粘贴（磁盘读取）] 预览文件读取失败 %s，原样返回' % src_path, flush=True)
            return (原图,)
        age = time.time() - os.stat(src_path).st_mtime
        print('[面部粘贴（磁盘读取）] 脸源=%s age=%.1fs mean=%.5f' % (
            src_path, age, crop_u8.mean() / 255.0), flush=True)

        # ---- 磁盘备份(证据链) ----
        try:
            ts = time.strftime('%Y%m%d_%H%M%S') + '_' + str(int(time.time() * 1000) % 1000)
            d = os.path.join(EVIDENCE_DIR, 'e2_' + ts)
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, 'crop.png'), cv2.cvtColor(crop_u8, cv2.COLOR_RGB2BGR))
            with open(os.path.join(d, 'src.txt'), 'w', encoding='utf-8') as f:
                f.write(src_path + '\n')
        except Exception as e:
            print('[面部粘贴（磁盘读取）] evidence err %r' % e, flush=True)

        # ---- 粘贴（与 face_paste.py 单张分支同逻辑，全图坐标遮罩）----
        paste_img = _img_to_uint8(原图).copy()
        H, W = paste_img.shape[:2]
        mask = _mask01(裁剪遮罩).copy()
        full_mask = (mask.shape[0] == H and mask.shape[1] == W)
        if not full_mask and mask.shape[:2] != crop_u8.shape[:2]:
            mask = cv2.resize(mask, (crop_u8.shape[1], crop_u8.shape[0]), interpolation=cv2.INTER_NEAREST)
        if rotated and angle != 0 and center is not None:
            face_resized = cv2.resize(crop_u8, (w, h))
            if not full_mask:
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            pad = int(np.hypot(W, H))
            padded = cv2.copyMakeBorder(paste_img.copy(), pad, pad, pad, pad, borderType=cv2.BORDER_REPLICATE)
            full_valid = np.ones((H + 2 * pad, W + 2 * pad), dtype=np.uint8) * 255
            center_p = (center[0] + pad, center[1] + pad)
            M_rot_p = cv2.getRotationMatrix2D(center_p, angle, 1)
            rotated_full_p = cv2.warpAffine(padded, M_rot_p, (W + 2 * pad, H + 2 * pad), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            rotated_valid_p = cv2.warpAffine(full_valid, M_rot_p, (W + 2 * pad, H + 2 * pad), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + w, W), min(y + h, H)
            if x2 > x1 and y2 > y1:
                sx = 0 if x >= 0 else -x
                sy = 0 if y >= 0 else -y
                sw = x2 - x1
                sh = y2 - y1
                patch = face_resized[sy:sy + sh, sx:sx + sw]
                if full_mask:
                    mask_bool = np.ones((sh, sw), dtype=bool)
                else:
                    mpatch = mask_resized[sy:sy + sh, sx:sx + sw]
                    mask_bool = (mpatch > 0.5)
                roi = rotated_full_p[y1 + pad:y2 + pad, x1 + pad:x2 + pad]
                roi[mask_bool] = patch[mask_bool]
                rotated_full_p[y1 + pad:y2 + pad, x1 + pad:x2 + pad] = roi
                rotated_valid_p[y1 + pad:y2 + pad, x1 + pad:x2 + pad][mask_bool] = 255
            M_back_p = cv2.getRotationMatrix2D(center_p, -angle, 1)
            inv_img_p = cv2.warpAffine(rotated_full_p, M_back_p, (W + 2 * pad, H + 2 * pad), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            inv_valid_p = cv2.warpAffine(rotated_valid_p, M_back_p, (W + 2 * pad, H + 2 * pad), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            inv_img = inv_img_p[pad:pad + H, pad:pad + W]
            inv_valid = inv_valid_p[pad:pad + H, pad:pad + W]
            valid_mask = (inv_valid > 127).astype(np.uint8)
            alpha = valid_mask[..., None].astype(np.float32)
            if full_mask:
                alpha = alpha * (mask > 0.5).astype(np.float32)[..., None]
            paste_img = (paste_img.astype(np.float32) * (1.0 - alpha) + inv_img.astype(np.float32) * alpha).astype(np.uint8)
        else:
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + w, W), min(y + h, H)
            target_w, target_h = x2 - x1, y2 - y1
            if target_w > 0 and target_h > 0:
                face_resized = cv2.resize(crop_u8, (target_w, target_h))
                if full_mask:
                    overlay = paste_img.copy()
                    overlay[y1:y2, x1:x2] = face_resized
                    mfull = (mask > 0.5).astype(np.uint8)
                    for c in range(3):
                        paste_img[..., c] = paste_img[..., c] * (1 - mfull) + overlay[..., c] * mfull
                else:
                    mask_resized = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                    roi = paste_img[y1:y2, x1:x2]
                    mask_bool = (mask_resized > 0.5).astype(np.uint8)
                    for c in range(3):
                        roi[..., c] = roi[..., c] * (1 - mask_bool) + face_resized[..., c] * mask_bool
                    paste_img[y1:y2, x1:x2] = roi
        t = torch.from_numpy(paste_img.astype(np.float32) / 255.0).contiguous().unsqueeze(0)
        print('[面部粘贴（磁盘读取）] DONE box=%s full_mask=%s' % (裁剪数据["box"], full_mask), flush=True)
        return (t,)


NODE_CLASS_MAPPINGS = {
    "面部粘贴（磁盘读取）": 面部粘贴磁盘读取,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "面部粘贴（磁盘读取）": "面部粘贴（磁盘读取）",
}
