import cv2
import numpy as np

# UniFace 替代 MediaPipe + DeepFace
try:
    from uniface import RetinaFace, FaceAnalyzer
    from uniface.attribute import AgeGender
except ImportError:
    RetinaFace = None
    print("[面部选择器] ❌ 未检测到 UniFace 库，请安装：pip install uniface")

class FaceAnalyzerUni:
    """用UniFace封装人脸检测+性别识别+角度计算"""
    def __init__(self):
        self.detector = None
        self.age_gender = None
        self.initialized = False
        try:
            if RetinaFace is None:
                print("[面部选择器] ❌ UniFace 未安装，请运行：pip install uniface")
                return
            self.detector = RetinaFace()
            self.age_gender = AgeGender()
            self.initialized = True
        except Exception as e:
            print(f"[面部选择器] ❌ UniFace 初始化失败: {e}")

    def detect_faces(self, image, min_size=50):
        """UniFace bbox 格式是 [x1,y1,x2,y2]，转换为 [x,y,w,h]
        只做检测；性别识别由 predict_genders 在置信度过滤之后执行，避免对低置信度脸浪费算力
        """
        if not self.initialized or self.detector is None:
            return []
        faces = []
        try:
            detections = self.detector.detect(image)
            for d in detections:
                bbox = np.array(d.bbox)
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox.astype(int)
                    bw = x2 - x1
                    bh = y2 - y1
                    if bw >= min_size and bh >= min_size:
                        landmarks = np.array(d.landmarks) if d.landmarks is not None else None
                        faces.append({
                            'box': [x1, y1, bw, bh],
                            'score': float(d.confidence),
                            'gender': 'unknown',
                            'landmarks': landmarks
                        })
        except Exception as e:
            print(f"[面部选择器] 检测异常: {e}")
        return faces

    def predict_genders(self, image, faces):
        """性别识别（应在置信度/尺寸过滤之后调用）
        image: 原图 BGR
        """
        if not self.initialized or self.age_gender is None or not faces:
            return faces
        class _Det:
            pass
        for f in faces:
            if f.get('gender', 'unknown') != 'unknown':
                continue
            try:
                x, y, w, h = [int(v) for v in f['box']]
                d = _Det()
                d.bbox = np.array([x, y, x + w, y + h])
                d.confidence = f.get('score', 1.0)
                d.landmarks = f.get('landmarks')
                result = self.age_gender.predict(image, d)
                if result.gender == 1:
                    f['gender'] = 'male'
                elif result.gender == 0:
                    f['gender'] = 'female'
            except Exception:
                f['gender'] = 'unknown'
        print(f"[面部选择器] 检测到的脸性别: {[f['gender'] for f in faces]}")
        return faces

    def get_face_angle(self, image, face):
        """用RetinaFace的5点关键点算旋转角度
        landmarks[0]=左眼, landmarks[1]=右眼
        ±5度以内算水平，不转
        """
        if not self.initialized:
            return 0
        landmarks = face.get('landmarks')
        if landmarks is not None and len(landmarks) >= 5:
            left_eye = (float(landmarks[0, 0]), float(landmarks[0, 1]))
            right_eye = (float(landmarks[1, 0]), float(landmarks[1, 1]))
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            if abs(dx) >= 5:
                angle = np.degrees(np.arctan2(dy, dx))
                if abs(angle) < 5:
                    return 0
                # dy>0:右眼偏低→cv2逆时针转=扶正
                # dy<0:右眼偏高→cv2顺时针转=扶正
                return angle
        return 0

    def sort_faces(self, faces, method='area'):
        if method == 'area':
            return sorted(faces, key=lambda f: f['box'][2]*f['box'][3], reverse=True)
        elif method == 'left':
            return sorted(faces, key=lambda f: f['box'][0])
        return faces

import comfy.sd

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def extract_mask_faces(mask, min_size=50, min_area=100):
    mask_bin = (mask > 127).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    faces = []
    return faces, contours


class 面部选择器:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "区分男女": (["不区分", "男", "女"], {"default": "不区分"}),
                "人物排序": (["像素占比", "从左向右"], {"default": "从左向右"}),
                "输出索引": ("STRING", {"default": "1", "multiline": False, "tooltip": "输入0输出全部，或如1 3、2,6、2，6、2。6等，支持空格/中英文逗号/句号分隔多个序号"}),
                "最小尺寸": ("INT", {"default": 50, "min": 0, "max": 5000}),
                "置信度阈值": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "裁剪系数": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0}),
                "是否旋转面部": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "辅助遮罩": ("MASK", ),
            }
        }

    RETURN_TYPES = ("IMAGE", "FACE_CROP_DATA", "MASK")
    RETURN_NAMES = ("裁剪图像", "裁剪数据", "裁剪遮罩")
    FUNCTION = "run"
    CATEGORY = "Muye/面部"

    def __init__(self):
        self.analyzer = FaceAnalyzerUni()

    def run(self, 图像, 区分男女, 人物排序, 输出索引, 最小尺寸, 置信度阈值, 裁剪系数, 是否旋转面部, 辅助遮罩=None):
        import numpy as np
        import torch
        import re

        # 1. 预处理原图
        if hasattr(图像, 'cpu') and hasattr(图像, 'numpy'):
            orig_img = 图像.cpu().numpy()
            if orig_img.ndim == 3 and orig_img.shape[0] in [1, 3, 4]:
                orig_img = np.transpose(orig_img, (1, 2, 0))
            if orig_img.ndim == 4 and orig_img.shape[0] == 1:
                orig_img = orig_img[0]
            orig_img = orig_img.copy()
            if orig_img.max() <= 1.0:
                orig_img = (orig_img * 255).clip(0, 255).astype(np.uint8)
            else:
                orig_img = orig_img.astype(np.uint8)
        else:
            orig_img = np.array(图像)
            if orig_img.ndim == 4 and orig_img.shape[0] == 1:
                orig_img = orig_img[0]
        img = orig_img
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        # 2. UniFace检测+置信度过滤+最小尺寸过滤（性别识别移到过滤之后，省算力）
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        faces_model = self.analyzer.detect_faces(img_bgr, min_size=最小尺寸)
        faces_model = [f for f in faces_model if f.get('score', 1.0) >= 置信度阈值]
        faces_model = [f for f in faces_model if f['box'][2] >= 最小尺寸 and f['box'][3] >= 最小尺寸]
        print(f"[面部选择器] UniFace检测到人脸数量: {len(faces_model)}")
        self.analyzer.predict_genders(img_bgr, faces_model)

        # 3. 遮罩box提取
        faces_mask = []
        crop_regions = []
        if 辅助遮罩 is not None:
            mask = 辅助遮罩
            if isinstance(mask, torch.Tensor):
                mask_np = mask.detach().cpu().numpy()
            else:
                mask_np = np.array(mask)
            mask_np = np.asarray(mask_np)
            mask_np = np.squeeze(mask_np)
            if mask_np.ndim != 2:
                raise ValueError(f"辅助遮罩格式错误，shape={mask_np.shape}，请提供单通道mask")
            if mask_np.max() <= 1.0:
                mask_np = (mask_np * 255).round().astype(np.uint8)
            else:
                mask_np = mask_np.astype(np.uint8)
            face_centers = []
            for f in faces_model:
                x, y, w, h = f['box']
                face_centers.append((x + w//2, y + h//2))
            _, contours = extract_mask_faces(mask_np, min_size=最小尺寸)
            try:
                from skimage.segmentation import watershed
                from scipy import ndimage as ndi
            except ImportError:
                watershed = None
            for cnt_i, cnt in enumerate(contours):
                x, y, w, h = cv2.boundingRect(cnt)
                region_centers = [pt for pt in face_centers if x <= pt[0] <= x+w and y <= pt[1] <= y+h]
                if len(region_centers) <= 1 or watershed is None:
                    if w >= 最小尺寸 and h >= 最小尺寸 and w*h >= 100:
                        faces_mask.append({'box': [x, y, w, h], 'region': cnt_i})
                else:
                    mask_region = np.zeros_like(mask_np)
                    cv2.drawContours(mask_region, [cnt], -1, 255, -1)
                    mask_region = (mask_region > 127).astype(np.uint8)
                    markers = np.zeros_like(mask_region, dtype=np.int32)
                    for i, pt in enumerate(region_centers):
                        cx, cy = int(pt[0]), int(pt[1])
                        if 0 <= cy < markers.shape[0] and 0 <= cx < markers.shape[1]:
                            markers[cy, cx] = i+1
                    distance = ndi.distance_transform_edt(mask_region)
                    labels = watershed(-distance, markers, mask=mask_region)
                    found_label = set()
                    for i in range(1, len(region_centers)+1):
                        mask_i = (labels == i).astype(np.uint8)
                        if np.sum(mask_i) == 0:
                            continue
                        found_label.add(i)
                        sub_contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for scnt in sub_contours:
                            xx, yy, ww, hh = cv2.boundingRect(scnt)
                            if ww >= 5 and hh >= 5 and ww*hh >= 10:
                                faces_mask.append({'box': [xx, yy, ww, hh], 'region': cnt_i})
                    for i, pt in enumerate(region_centers):
                        if (i+1) not in found_label:
                            cx, cy = int(pt[0]), int(pt[1])
                            r = max(最小尺寸//2, 10)
                            xx = max(cx - r, 0)
                            yy = max(cy - r, 0)
                            ww = min(r*2, mask_np.shape[1]-xx)
                            hh = min(r*2, mask_np.shape[0]-yy)
                            faces_mask.append({'box': [xx, yy, ww, hh], 'region': cnt_i})
            # 3a. 裁剪区域 = contour包围盒 ∪ 其中包含的模型脸box
            #     模型漏检的脸（分数过低被滤掉）对应区域里没有脸 → 区域仅含contour，按用户给的遮罩原样裁剪输出
            for cnt_i, cnt in enumerate(contours):
                x, y, w, h = cv2.boundingRect(cnt)
                rx1, ry1, rx2, ry2 = x, y, x + w, y + h
                for m in faces_mask:
                    if m.get('region') != cnt_i:
                        continue
                    bx, by, bw, bh = m['box']
                    rx1 = min(rx1, bx)
                    ry1 = min(ry1, by)
                    rx2 = max(rx2, bx + bw)
                    ry2 = max(ry2, by + bh)
                crop_regions.append([rx1, ry1, rx2 - rx1, ry2 - ry1])
            print(f"[面部选择器] 遮罩裁剪区域数: {len(crop_regions)}")
            print(f"[面部选择器] 遮罩检测到人脸数量: {len(faces_mask)}")

        # 4. IoU配对
        used_mask_idx = set()
        faces_final = []
        for i, f in enumerate(faces_model):
            best_iou = 0
            best_idx = -1
            for j, m in enumerate(faces_mask):
                if j in used_mask_idx:
                    continue
                iou = compute_iou(f['box'], m['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            if best_iou > 0.5 and best_idx >= 0:
                box = faces_mask[best_idx]['box']
                used_mask_idx.add(best_idx)
                from_mask = True
            else:
                box = f['box']
                from_mask = False
            faces_final.append({**f, 'box': box, 'crop_region': None, 'from_mask': from_mask})
        for j, m in enumerate(faces_mask):
            if j not in used_mask_idx:
                region = m.get('region')
                crop_region = crop_regions[region] if (region is not None and region < len(crop_regions)) else m['box']
                faces_final.append({
                    'box': m['box'], 'score': 1.0, 'gender': 'unknown', 'landmarks': None,
                    'crop_region': crop_region, 'from_mask': True
                })

        # 5. 性别识别已在 predict_genders（置信度过滤后）完成
        faces_before_gender = list(faces_final)

        # 6. 性别过滤（遮罩来源的脸豁免过滤：辅助遮罩是用户指定的脸集合，模型漏检的脸没有性别信息）
        if 区分男女 == "男":
            faces_final = [f for f in faces_final if f.get('from_mask') or f.get('gender', 'unknown') == 'male']
        elif 区分男女 == "女":
            faces_final = [f for f in faces_final if f.get('from_mask') or f.get('gender', 'unknown') == 'female']
        if (区分男女 in ("男", "女")) and (not faces_final):
            faces_final = faces_before_gender
            print(f"[面部选择器] 按性别过滤后无匹配，已回退到不区分性别的检测结果，共{len(faces_final)}个候选")
        elif (区分男女 in ("男", "女")) and (len(faces_final) < len(faces_before_gender)):
            kept_mask = sum(1 for f in faces_final if f.get('from_mask') and f.get('gender', 'unknown') == 'unknown')
            if kept_mask:
                print(f"[面部选择器] 提示: {kept_mask}张遮罩脸无模型性别信息，按遮罩保留未被过滤")

        # 7. 排序
        if 人物排序 == "像素占比":
            faces_final = self.analyzer.sort_faces(faces_final, method='area')
        else:
            faces_final = self.analyzer.sort_faces(faces_final, method='left')

        # 8. 解析输出索引
        idx_str = str(输出索引).strip()
        idx_raw_list = re.split(r'[\s,，。\.]+', idx_str)
        idx_raw_list = [s for s in idx_raw_list if s]
        idx_list = [int(s) for s in idx_raw_list if s.isdigit()]

        # 9. 输出辅助函数
        def np2torch(img_np, batch=False):
            arr = np.array(img_np)
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            if arr.ndim == 3 and arr.shape[2] > 3:
                arr = arr[:, :, :3]
            while arr.ndim > 3:
                arr = arr[0]
            arr = arr.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            tensor = torch.from_numpy(arr).contiguous()
            if batch:
                return tensor.unsqueeze(0)
            else:
                return tensor

        def expand_box(box, scale, img_shape):
            x, y, w, h = box
            cx, cy = x + w//2, y + h//2
            size = int(max(w, h) * scale)
            nx, ny = max(cx - size//2, 0), max(cy - size//2, 0)
            ex, ey = min(cx + size//2, img_shape[1]), min(cy + size//2, img_shape[0])
            return [nx, ny, ex-nx, ey-ny]

        def mask_full(box=None):
            """输出遮罩统一为 ComfyUI MASK 规范：float32、0-1、全图尺寸（与 BBox 检测器输出同构，可直连预览）
            返回 2D [H,W] tensor；调用方按需 unsqueeze 成 [1,H,W]（单张）或 stack 成 [N,H,W]（多张）
            box: 原图坐标 [x,y,w,h]，在裁剪框位置画白；None=全白（无脸兜底）
            """
            H, W = orig_img.shape[:2]
            m = np.zeros((H, W), dtype=np.float32)
            if box is None:
                m[:] = 1.0  # 无脸兜底：全白（与原设计一致）
            else:
                bx, by, bw, bh = box
                x1, y1 = max(int(bx), 0), max(int(by), 0)
                x2, y2 = min(int(bx) + int(bw), W), min(int(by) + int(bh), H)
                if x2 > x1 and y2 > y1:
                    m[y1:y2, x1:x2] = 1.0
            return torch.from_numpy(m.copy()).contiguous()

        def make_crop(face):
            """非旋转裁剪：以裁剪区域（有辅助遮罩时为遮罩区域）为中心扩展裁剪，白满矩形遮罩"""
            box = face.get('crop_region') or face['box']
            x, y, w, h = expand_box(box, 裁剪系数, orig_img.shape)
            crop_img = orig_img[y:y+h, x:x+w].copy()
            mask = np.zeros(orig_img.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
            mask_crop = mask[y:y+h, x:x+w].copy()
            center = (x + w/2, y + h/2)
            crop_data = {'box': [x, y, w, h], 'angle': 0, 'center': center, 'rotated': False}
            return crop_img, mask_crop, crop_data

        if not faces_final:
            return (np2torch(orig_img, batch=True), {"box": None, "angle": 0}, mask_full(None).unsqueeze(0))

        # 10. 旋转逻辑：以人脸box中心为旋转中心
        def rotate_box_from_full(img_full, face_orig_box, angle):
            H, W = img_full.shape[:2]
            h0, w0 = H, W
            fx, fy, fw, fh = face_orig_box
            center = (fx + fw/2, fy + fh/2)
            M = cv2.getRotationMatrix2D(center, angle, 1)
            rotated_img = cv2.warpAffine(img_full, M, (w0, h0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            ccx = int(fx + fw/2)
            ccy = int(fy + fh/2)
            size = int(max(fw, fh) * 裁剪系数)
            nnx, nny = max(ccx - size//2, 0), max(ccy - size//2, 0)
            nex, ney = min(ccx + size//2, w0), min(ccy + size//2, h0)
            crop_img = rotated_img[nny:ney, nnx:nex].copy()
            mask_crop_full = np.zeros((h0, w0), dtype=np.uint8)
            cv2.rectangle(mask_crop_full, (nnx, nny), (nex, ney), 255, -1)
            mask_crop = mask_crop_full[nny:ney, nnx:nex].copy()
            crop_data = {
                'box': [nnx, nny, nex-nnx, ney-nny],
                'angle': angle,
                'center': center,
                'rotated': True
            }
            return crop_img, mask_crop, crop_data

        # 排序方式名称
        sort_label = "从左至右" if 人物排序 == "从左向右" else "像素占比"
        total_faces = len(faces_final)

        # 判定逻辑：0 = 输出全部
        if 0 in idx_list:
            if not faces_final:
                return (np2torch(orig_img, batch=True), {"box": None, "angle": 0}, mask_full(None).unsqueeze(0))
            crop_imgs, crop_datas = [], []
            rotated_info = []
            for idx_num, face in enumerate(faces_final):
                if 是否旋转面部:
                    try:
                        angle = self.analyzer.get_face_angle(img, face)
                    except Exception:
                        angle = 0
                    crop_img, mask_crop, crop_data = rotate_box_from_full(orig_img, face['box'], angle)
                    if angle != 0:
                        rotated_info.append(f"#{idx_num+1} {angle:+.1f}°")
                else:
                    crop_img, mask_crop, crop_data = make_crop(face)
                crop_imgs.append(np2torch(crop_img, batch=False))
                crop_datas.append(crop_data)
            # 输出日志
            output_info = f"[面部选择器] 输出 {total_faces}/{total_faces} {sort_label} 全部"
            if rotated_info:
                output_info += f" | 旋转: {', '.join(rotated_info)}"
            print(output_info)
            # IMAGE: list 批量（各脸 crop 尺寸可不同，ComfyUI 原生格式）
            # 遮罩: 全图坐标 [N,H,W] float32 0-1（MASK 规范，与 BBox 检测器同构，可直连预览）
            mask_out = torch.stack([mask_full(d['box']) for d in crop_datas], dim=0)
            return (tuple(crop_imgs), crop_datas, mask_out)

        # 多序号输出
        valid_idxs = [idx for idx in idx_list if 1 <= idx <= len(faces_final)]
        if idx_list and (not valid_idxs):
            print(f"[面部选择器] [警告] 输出索引{idx_list}全部越界（共检测到{len(faces_final)}张脸），回退输出整图")
        elif len(valid_idxs) < len(idx_list):
            invalid = sorted(set(idx_list) - set(valid_idxs))
            print(f"[面部选择器] [警告] 输出索引含越界序号{invalid}，仅输出有效索引{valid_idxs}")
        if len(valid_idxs) == 1:
            idx = valid_idxs[0] - 1
            face = faces_final[idx]
            angle_log = ""
            if 是否旋转面部:
                try:
                    angle = self.analyzer.get_face_angle(img, face)
                except Exception:
                    angle = 0
                crop_img, mask_crop, crop_data = rotate_box_from_full(orig_img, face['box'], angle)
                if angle != 0:
                    angle_log = f" | 旋转 #{valid_idxs[0]} {angle:+.1f}°"
            else:
                crop_img, mask_crop, crop_data = make_crop(face)
            print(f"[面部选择器] 输出 {len(valid_idxs)}/{total_faces} {sort_label} {valid_idxs}{angle_log}")
            return (np2torch(crop_img, batch=True), crop_data, mask_full(crop_data['box']).unsqueeze(0))
        elif len(valid_idxs) > 1:
            crop_imgs, crop_datas = [], []
            rotated_info = []
            for idx in valid_idxs:
                i = idx - 1
                face = faces_final[i]
                if 是否旋转面部:
                    try:
                        angle = self.analyzer.get_face_angle(img, face)
                    except Exception:
                        angle = 0
                    crop_img, mask_crop, crop_data = rotate_box_from_full(orig_img, face['box'], angle)
                    if angle != 0:
                        rotated_info.append(f"#{idx} {angle:+.1f}°")
                else:
                    crop_img, mask_crop, crop_data = make_crop(face)
                crop_imgs.append(np2torch(crop_img, batch=False))
                crop_datas.append(crop_data)
            output_info = f"[面部选择器] 输出 {len(valid_idxs)}/{total_faces} {sort_label} {valid_idxs}"
            if rotated_info:
                output_info += f" | 旋转: {', '.join(rotated_info)}"
            print(output_info)
            # IMAGE: list 批量（各脸 crop 尺寸可不同，ComfyUI 原生格式）
            # 遮罩: 全图坐标 [N,H,W] float32 0-1（MASK 规范，与 BBox 检测器同构，可直连预览）
            mask_out = torch.stack([mask_full(d['box']) for d in crop_datas], dim=0)
            return (tuple(crop_imgs), crop_datas, mask_out)
        else:
            return (np2torch(orig_img, batch=True), {"box": None, "angle": 0}, mask_full(None).unsqueeze(0))

# 节点注册导出（名称改为"面部选择器"，原高级节点的class名也改）
NODE_CLASS_MAPPINGS = {
    "面部选择器": 面部选择器,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "面部选择器": "面部选择器",
}