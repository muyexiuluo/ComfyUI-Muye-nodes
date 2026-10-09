"""多图宫格拼接（MuyeImageList / MuyeGridImageConcat）。

节点说明：
- 图像列表：把多路图像收集成一个列表（最多 16 路，未接输入自动跳过）。
- 宫格图像拼接：把列表里的 N 张图拼成最方正的宫格（1-16 张，最多 4 列），自带预览。
  自动剔除全黑占位帧（上游批次类节点对未接输入常补黑图占位，只拼实际有图的席位）。
  多图像端口的类型一律用标准 IMAGE（前端 LiteGraph 连线校验不认自定义新类型）。
"""
import os
import math
import random

import torch
import numpy as np
import folder_paths
from PIL import Image
from torch.nn.functional import interpolate

MAX_IMAGES = 16
MAX_COLS = 4


def _is_empty_frame(img):
    """全零帧检测：上游批次类节点（如 KJNodes 图像组合批次）对未接输入会补 torch.zeros
    黑图占位。实际图像几乎不可能是纯零帧。返回 True 表示是空席位。"""
    t = img.detach().to(device="cpu").float().view(-1)
    return t.numel() == 0 or t.max().item() == 0.0


def _norm_image(img):
    """把输入统一成 [1,H,W,3] float32 0-1 CPU 张量（取 batch 第 0 张）。
    兼容: [B,H,W,C] 图像 / [H,W] 单通道遮罩 / [B,H,W] 单通道 / [H,W,C] 单张
    """
    if isinstance(img, (list, tuple)):
        img = img[0]
    img = img.detach().to(device="cpu").float()
    if img.dim() == 2:          # [H,W] 遮罩
        img = img.unsqueeze(0).unsqueeze(-1)   # → [1,H,W,1]
    elif img.dim() == 3:        # [B,H,W] 单通道 或 [H,W,C] 单张
        if img.shape[0] == 1 or img.shape[-1] > 1:
            img = img.unsqueeze(0)             # [H,W,C] → [1,H,W,C]
        else:
            img = img.unsqueeze(-1)            # [B,H,W] → [B,H,W,1]
    if img.shape[0] > 1:
        img = img[0:1]
    c = img.shape[-1]
    if c == 1:
        img = img.repeat(1, 1, 1, 3)
    elif c > 3:
        img = img[..., 0:3]
    img = img.clamp(0.0, 1.0).contiguous()
    return img


def _resize(img, w, h):
    if img.shape[1] == h and img.shape[2] == w:
        return img
    x = img.permute(0, 3, 1, 2)
    x = interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).contiguous()


def make_grid(images, cols, rows, match_size, gap, bg):
    """images: 已规范化的 [1,H,W,3] 列表，按行优先摆进 cols x rows 网格，返回 [1,H,W,3]"""
    n = len(images)
    if match_size:
        cell_w, cell_h = images[0].shape[2], images[0].shape[1]
        # 等比适配：不拉伸不变形，留白由背景色填充
        sizes = []
        for i in range(n):
            w, h = images[i].shape[2], images[i].shape[1]
            s = min(cell_w / w, cell_h / h)
            sizes.append((max(1, int(round(w * s))), max(1, int(round(h * s)))))
        col_ws = [cell_w] * cols
        row_hs = [cell_h] * rows
    else:
        col_ws = []
        for c in range(cols):
            idxs = [r * cols + c for r in range(rows) if r * cols + c < n]
            col_ws.append(max(images[i].shape[2] for i in idxs))
        row_hs = []
        for r in range(rows):
            idxs = [i for i in range(r * cols, min((r + 1) * cols, n))]
            row_hs.append(max(images[i].shape[1] for i in idxs))
        sizes = []
        for i in range(n):
            r, c = divmod(i, cols)
            sizes.append((col_ws[c], row_hs[r]))

    canvas_w = sum(col_ws) + (cols - 1) * gap
    canvas_h = sum(row_hs) + (rows - 1) * gap
    bgv = 0.0 if bg == "黑色" else 1.0
    canvas = torch.full((1, canvas_h, canvas_w, 3), float(bgv), dtype=torch.float32)

    xs = []
    x = 0
    for w in col_ws:
        xs.append(x)
        x += w + gap
    ys = []
    y = 0
    for h in row_hs:
        ys.append(y)
        y += h + gap

    for i in range(n):
        r, c = divmod(i, cols)
        cw, ch = sizes[i]
        img = _resize(images[i], cw, ch)
        x0 = xs[c] + (col_ws[c] - cw) // 2
        y0 = ys[r] + (row_hs[r] - ch) // 2
        canvas[:, y0:y0 + ch, x0:x0 + cw, :] = img
    return canvas


class 图像列表:
    """把多路图像收集成一个列表，供 宫格图像拼接 等节点使用。
    从 图像_1 开始按顺序连接，未连接的输入自动跳过；最多 16 路。
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, MAX_IMAGES + 1):
            optional[f"图像_{i}"] = (["IMAGE"], {"forceInput": True,
                                                  "tooltip": f"第 {i} 路图像（可选，不接自动跳过）"})
        return {"optional": optional}

    # 输出端口用标准 IMAGE 类型（数据可以是单张 [1,H,W,C] 或张量列表）：
    # LiteGraph 前端连线校验不认自定义新类型，标准 IMAGE 前后端都通。
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "run"
    CATEGORY = "Muye/图像"

    def run(self, **kwargs):
        images = []
        for i in range(1, MAX_IMAGES + 1):
            v = kwargs.get(f"图像_{i}")
            if v is not None:
                images.append(_norm_image(v))
        if not images:
            raise ValueError("图像列表: 没有连接到任何图像输入")
        print(f"[图像列表] 收集到 {len(images)} 张图像")
        return (images,)


class 宫格图像拼接:
    """把一张列表里的 N 张图拼成最方正的宫格（最多 16 张 / 4x4：
    1张→1x1，2张→2x1，3-4张→2x2，5-9张→3列，10-16张→4列）。
    自动剔除全黑占位帧（KJNodes 图像组合批次 对未接输入补黑图占位，只拼实际有图的席位）。
    匹配尺寸=开时所有格统一成第一张图的尺寸、图在格内等比适配居中（不裁剪不变形，留白填背景色）；关时各图保持原尺寸、在格内居中。
    本节点自带预览（节点本体出图），同时输出真实 IMAGE 可继续往下接。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE", {"forceInput": True,
                                  "tooltip": "支持 4-D batch / 张量列表 / 单张，自动剔除上游补的黑图占位帧"}),
                "匹配尺寸": ("BOOLEAN", {"default": True,
                                         "tooltip": "开=所有格统一成第一张图尺寸；关=各图保持原尺寸居中"}),
                "间隙": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1,
                                 "tooltip": "格子之间的像素间隙"}),
                "背景色": (["黑色", "白色"], {"default": "黑色", "tooltip": "间隙与空格填充色"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Muye/图像"

    def run(self, 图像, 匹配尺寸=True, 间隙=0, 背景色="黑色"):
        # 对齐「图像列表（批次）统一尺寸」的输入处理：batch tensor 按 batch 维拆，列表逐张，单张当 1 张
        if isinstance(图像, torch.Tensor):
            if 图像.ndim == 4:
                items = [图像[i] for i in range(图像.shape[0])]
            elif 图像.ndim == 3:
                items = [图像]
            else:
                raise ValueError("宫格图像拼接: 图像 输入格式不支持（需要 3-D/4-D 张量或张量列表）")
        elif isinstance(图像, (list, tuple)):
            items = list(图像)
        else:
            raise ValueError("宫格图像拼接: 图像 输入格式不支持（需要 3-D/4-D 张量或张量列表）")
        items = [it for it in items if it is not None]
        if not items:
            raise ValueError("宫格图像拼接: 列表里没有图像")
        kept, dropped = [], 0
        for it in items:
            if _is_empty_frame(it):
                dropped += 1
            else:
                kept.append(it)
        if dropped:
            print(f"[宫格图像拼接] 剔除 {dropped} 张全黑占位帧，剩余 {len(kept)} 张")
        items = kept
        if not items:
            raise ValueError("宫格图像拼接: 剔除空席位后没有实际图像")
        if len(items) > MAX_IMAGES:
            print(f"[宫格图像拼接] 警告: 列表共 {len(items)} 张，最多支持 {MAX_IMAGES} 张，已截取前 {MAX_IMAGES} 张")
        images = [_norm_image(it) for it in items[:MAX_IMAGES]]

        cols = min(MAX_COLS, math.ceil(math.sqrt(len(images))))
        rows = math.ceil(len(images) / cols)
        grid = make_grid(images, cols, rows, 匹配尺寸, 间隙, 背景色)
        print(f"[宫格图像拼接] {len(images)} 张 → {cols}x{rows} 宫格，输出 {grid.shape[2]}x{grid.shape[1]}")
        ui = {"images": self._preview(grid)}
        return {"ui": ui, "result": (grid,)}

    def _preview(self, image):
        result = []
        out_dir = folder_paths.get_temp_directory()
        prefix = "MuyeGrid_temp_" + "".join(random.choice("abcdefghijklmnopqrstuvxyz") for _ in range(5))
        filename = f"{prefix}.png"
        file_path = os.path.join(out_dir, filename)
        img = 255.0 * image[0].cpu().numpy()
        img = np.clip(img, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(file_path, compress_level=1)
        result.append({"filename": filename, "subfolder": "", "type": "temp"})
        return result


NODE_CLASS_MAPPINGS = {
    "MuyeImageList": 图像列表,
    "MuyeGridImageConcat": 宫格图像拼接,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MuyeImageList": "图像列表",
    "MuyeGridImageConcat": "宫格图像拼接",
}
