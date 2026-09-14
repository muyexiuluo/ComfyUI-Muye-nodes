import cv2
import numpy as np
import torch

def to_mask01(m):
    """遮罩兼容：tensor/ndarray、float 0-1 或 uint8 0-255 → float32 0-1 2D"""
    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().numpy()
    arr = np.asarray(m)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        # 防御：混入 RGB 遮罩/单张 4D 残留，取第一通道
        arr = arr[0] if arr.shape[0] == 1 else arr[..., 0]
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr

class 面部粘贴:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "原图": ("IMAGE",),
                "裁剪图像": ("IMAGE",),
                "裁剪数据": ("FACE_CROP_DATA",),
                "裁剪遮罩": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("输出图像",)
    FUNCTION = "run"
    CATEGORY = "Muye/面部"

    def run(self, 原图, 裁剪图像, 裁剪数据, 裁剪遮罩):
        import collections.abc
        def is_list(obj):
            return isinstance(obj, (list, tuple))

        def np2torch(img_np):
            arr = np.array(img_np)
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            if arr.ndim == 3 and arr.shape[2] > 3:
                arr = arr[:, :, :3]
            arr = arr.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            tensor = torch.from_numpy(arr).contiguous()
            return tensor.unsqueeze(0)

        def to_numpy(img):
            import numbers
            if isinstance(img, torch.Tensor):
                arr = img.detach().cpu().numpy()
                if arr.ndim == 4 and arr.shape[0] == 1:
                    arr = arr[0]
                    # CHW 兼容：(1,3,H,W) → (H,W,3)，避免与 HWC 混淆后静默截断通道
                    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
                        arr = np.transpose(arr, (1, 2, 0))
            else:
                arr = np.array(img)
            while arr.ndim > 3:
                arr = arr.squeeze(0)
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            if arr.ndim == 3 and arr.shape[2] > 3:
                arr = arr[:, :, :3]
            if not (arr.dtype == np.uint8 or arr.dtype == np.float32 or arr.dtype == np.float64):
                arr = arr.astype(np.float32)
            if arr.max() == arr.min():
                # 均匀图保护：固定归一到灰 128（旧逻辑 /arr.max() 会把均匀图变成全白/全亮）
                arr = np.zeros_like(arr) + 128
            if arr.max() > 1.0:
                # 0-255 → 0-1：固定除以 255（旧逻辑 /arr.max() 会把暗部照片整体提亮）
                arr = arr / 255.0
            arr = np.clip(arr, 0, 1)
            arr = (arr * 255).round().astype(np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            if arr.ndim == 3 and arr.shape[2] > 3:
                arr = arr[:, :, :3]
            return arr

        # 批量粘贴（遮罩可为 [N,H,W] 全图坐标 tensor，zip 按 dim0 逐张遍历）
        if is_list(裁剪图像) and is_list(裁剪数据):
            paste_img = to_numpy(原图).copy()
            for face_img, crop_data, mask in zip(裁剪图像, 裁剪数据, 裁剪遮罩):
                if not crop_data or not (isinstance(crop_data, dict) and crop_data.get("box")):
                    continue
                x, y, w, h = crop_data["box"]
                angle = crop_data.get("angle", 0)
                center = crop_data.get("center", None)
                rotated = crop_data.get("rotated", False)
                face_img_np = to_numpy(face_img).copy()
                H, W = paste_img.shape[:2]
                mask_np = to_mask01(mask).copy()
                # 全图坐标遮罩（[H,W]，与 BBox 检测器/面部选择器输出同款）：直接按原图坐标混合
                full_mask = (mask_np.shape[0] == H and mask_np.shape[1] == W)
                if not full_mask:
                    # 兼容旧格式：crop 尺寸遮罩，先对齐到 crop 真实尺寸
                    if mask_np.shape[:2] != (face_img_np.shape[0], face_img_np.shape[1]):
                        mask_np = cv2.resize(mask_np, (face_img_np.shape[1], face_img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
                if rotated and angle != 0 and center is not None:
                    face_resized = cv2.resize(face_img_np, (w, h))
                    if not full_mask:
                        mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                    pad = int(np.hypot(W, H))
                    padded = cv2.copyMakeBorder(paste_img.copy(), pad, pad, pad, pad, borderType=cv2.BORDER_REPLICATE)
                    full_valid = np.ones((H + 2*pad, W + 2*pad), dtype=np.uint8) * 255
                    center_p = (center[0] + pad, center[1] + pad)
                    M_rot_p = cv2.getRotationMatrix2D(center_p, angle, 1)
                    rotated_full_p = cv2.warpAffine(padded, M_rot_p, (W + 2*pad, H + 2*pad), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    rotated_valid_p = cv2.warpAffine(full_valid, M_rot_p, (W + 2*pad, H + 2*pad), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    x1, y1 = max(x, 0), max(y, 0)
                    x2, y2 = min(x + w, W), min(y + h, H)
                    if x2 > x1 and y2 > y1:
                        sx = 0 if x >= 0 else -x
                        sy = 0 if y >= 0 else -y
                        sw = x2 - x1
                        sh = y2 - y1
                        patch = face_resized[sy:sy+sh, sx:sx+sw]
                        if full_mask:
                            # 全图遮罩模式：box 区域整体粘贴，最终按全图遮罩混合
                            mask_bool = np.ones((sh, sw), dtype=bool)
                        else:
                            mpatch = mask_resized[sy:sy+sh, sx:sx+sw]
                            mask_bool = (mpatch > 0.5)
                        roi = rotated_full_p[y1+pad:y2+pad, x1+pad:x2+pad]
                        roi[mask_bool] = patch[mask_bool]
                        rotated_full_p[y1+pad:y2+pad, x1+pad:x2+pad] = roi
                        rotated_valid_p[y1+pad:y2+pad, x1+pad:x2+pad][mask_bool] = 255
                    M_back_p = cv2.getRotationMatrix2D(center_p, -angle, 1)
                    inv_img_p = cv2.warpAffine(rotated_full_p, M_back_p, (W + 2*pad, H + 2*pad), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    inv_valid_p = cv2.warpAffine(rotated_valid_p, M_back_p, (W + 2*pad, H + 2*pad), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    inv_img = inv_img_p[pad:pad+H, pad:pad+W]
                    inv_valid = inv_valid_p[pad:pad+H, pad:pad+W]
                    valid_mask = (inv_valid > 127).astype(np.uint8)
                    alpha = valid_mask[..., None].astype(np.float32)
                    if full_mask:
                        alpha = alpha * (mask_np > 0.5).astype(np.float32)[..., None]
                    paste_img = (paste_img.astype(np.float32) * (1.0 - alpha) + inv_img.astype(np.float32) * alpha).astype(np.uint8)
                else:
                    x1, y1 = max(x, 0), max(y, 0)
                    x2, y2 = min(x+w, W), min(y+h, H)
                    target_w, target_h = x2-x1, y2-y1
                    if target_w > 0 and target_h > 0:
                        face_resized = cv2.resize(face_img_np, (target_w, target_h))
                        if full_mask:
                            # 全图遮罩模式：box 位置放 crop，再按全图遮罩混合
                            overlay = paste_img.copy()
                            overlay[y1:y2, x1:x2] = face_resized
                            mfull = (mask_np > 0.5).astype(np.uint8)
                            for c in range(3):
                                paste_img[..., c] = paste_img[..., c] * (1 - mfull) + overlay[..., c] * mfull
                        else:
                            mask_resized = cv2.resize(mask_np, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                            roi = paste_img[y1:y2, x1:x2]
                            mask_bool = (mask_resized > 0.5).astype(np.uint8)
                            for c in range(3):
                                roi[...,c] = roi[...,c] * (1-mask_bool) + face_resized[...,c] * mask_bool
                            paste_img[y1:y2, x1:x2] = roi
            return (np2torch(paste_img),)
        # 单张粘贴
        elif isinstance(裁剪数据, dict):
            if not 裁剪数据 or not 裁剪数据.get("box"):
                return (裁剪图像,)
            x, y, w, h = 裁剪数据["box"]
            angle = 裁剪数据.get("angle", 0)
            center = 裁剪数据.get("center", None)
            rotated = 裁剪数据.get("rotated", False)
            paste_img = to_numpy(原图).copy()
            face_img = to_numpy(裁剪图像).copy()
            H, W = paste_img.shape[:2]
            mask = to_mask01(裁剪遮罩).copy()
            # 全图坐标遮罩（[H,W]，与 BBox 检测器/面部选择器输出同款）：直接按原图坐标混合
            full_mask = (mask.shape[0] == H and mask.shape[1] == W)
            if not full_mask:
                # 兼容旧格式：crop 尺寸遮罩，先对齐到 crop 真实尺寸
                if mask.shape[:2] != (face_img.shape[0], face_img.shape[1]):
                    mask = cv2.resize(mask, (face_img.shape[1], face_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            if rotated and angle != 0 and center is not None:
                face_resized = cv2.resize(face_img, (w, h))
                if not full_mask:
                    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                pad = int(np.hypot(W, H))
                padded = cv2.copyMakeBorder(paste_img.copy(), pad, pad, pad, pad, borderType=cv2.BORDER_REPLICATE)
                full_valid = np.ones((H + 2*pad, W + 2*pad), dtype=np.uint8) * 255
                center_p = (center[0] + pad, center[1] + pad)
                M_rot_p = cv2.getRotationMatrix2D(center_p, angle, 1)
                rotated_full_p = cv2.warpAffine(padded, M_rot_p, (W + 2*pad, H + 2*pad), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                rotated_valid_p = cv2.warpAffine(full_valid, M_rot_p, (W + 2*pad, H + 2*pad), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                x1, y1 = max(x, 0), max(y, 0)
                x2, y2 = min(x + w, W), min(y + h, H)
                if x2 > x1 and y2 > y1:
                    sx = 0 if x >= 0 else -x
                    sy = 0 if y >= 0 else -y
                    sw = x2 - x1
                    sh = y2 - y1
                    patch = face_resized[sy:sy+sh, sx:sx+sw]
                    if full_mask:
                        # 全图遮罩模式：box 区域整体粘贴，最终按全图遮罩混合
                        mask_bool = np.ones((sh, sw), dtype=bool)
                    else:
                        mpatch = mask_resized[sy:sy+sh, sx:sx+sw]
                        mask_bool = (mpatch > 0.5)
                    roi = rotated_full_p[y1+pad:y2+pad, x1+pad:x2+pad]
                    roi[mask_bool] = patch[mask_bool]
                    rotated_full_p[y1+pad:y2+pad, x1+pad:x2+pad] = roi
                    rotated_valid_p[y1+pad:y2+pad, x1+pad:x2+pad][mask_bool] = 255
                M_back_p = cv2.getRotationMatrix2D(center_p, -angle, 1)
                inv_img_p = cv2.warpAffine(rotated_full_p, M_back_p, (W + 2*pad, H + 2*pad), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                inv_valid_p = cv2.warpAffine(rotated_valid_p, M_back_p, (W + 2*pad, H + 2*pad), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                inv_img = inv_img_p[pad:pad+H, pad:pad+W]
                inv_valid = inv_valid_p[pad:pad+H, pad:pad+W]
                valid_mask = (inv_valid > 127).astype(np.uint8)
                alpha = valid_mask[..., None].astype(np.float32)
                if full_mask:
                    alpha = alpha * (mask > 0.5).astype(np.float32)[..., None]
                paste_img = (paste_img.astype(np.float32) * (1.0 - alpha) + inv_img.astype(np.float32) * alpha).astype(np.uint8)
            else:
                x1, y1 = max(x, 0), max(y, 0)
                x2, y2 = min(x+w, W), min(y+h, H)
                target_w, target_h = x2-x1, y2-y1
                if target_w > 0 and target_h > 0:
                    face_resized = cv2.resize(face_img, (target_w, target_h))
                    if full_mask:
                        # 全图遮罩模式：box 位置放 crop，再按全图遮罩混合
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
                            roi[...,c] = roi[...,c] * (1-mask_bool) + face_resized[...,c] * mask_bool
                        paste_img[y1:y2, x1:x2] = roi
            return (np2torch(paste_img),)
        else:
            print("[面部粘贴] [警告] 输入格式无法识别（裁剪数据既不是list也不是dict），原图原样返回！请检查面部选择器与面部粘贴版本是否配套")
            return (np2torch(to_numpy(原图)),)

# 节点注册导出
NODE_CLASS_MAPPINGS = {
    "面部粘贴": 面部粘贴,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "面部粘贴": "面部粘贴",
}