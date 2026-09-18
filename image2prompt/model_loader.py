import torch
import json
import os
import gc

from transformers import AutoProcessor, BitsAndBytesConfig

# ════════════════════════════════════════════════════════════
CAPTION_MODEL = "Muye_CaptionModel"


class CaptionModelWrapper:
    # 类级别缓存：记录所有创建的实例，方便统一卸载
    _instances = []

    def __init__(self, model, processor, arch_type, model_path, model_name,
                 quantization="bf16", attention="eager", gpu_choice="显卡1"):
        self.model = model
        self.processor = processor
        self.arch_type = arch_type
        self.model_path = model_path
        self.model_name = model_name
        self.quantization = quantization
        self.attention = attention
        self.gpu_choice = gpu_choice
        self._is_loaded = True
        CaptionModelWrapper._instances.append(self)

    def __repr__(self):
        return f"<CaptionModel: {self.model_name} ({self.arch_type})>"

    def reload(self):
        """重新加载模型（被 unload_all 清理后自动调用）"""
        if self._is_loaded and self.model is not None:
            return True
        if not self.model_path or not os.path.isdir(self.model_path):
            print(f"[木叶·图像反推] 警告: 找不到模型路径 '{self.model_path}'")
            return False
        model, processor, arch_type = load_model_with_options(
            self.model_path, self.quantization, self.attention,
            self.gpu_choice,
        )
        if model is None:
            print(f"[木叶·图像反推] 重新加载失败: {self.model_name}")
            return False
        self.model = model
        self.processor = processor
        self.arch_type = arch_type
        self._is_loaded = True
        print(f"[木叶·图像反推] 模型已自动重新加载: {self}")
        return True

    @classmethod
    def unload_all(cls):
        """卸载所有 Caption 模型的 GPU 权重，释放显存（保留实例和配置）"""
        freed = 0
        for wrapper in cls._instances:
            if not wrapper._is_loaded:
                continue
            if wrapper.model is not None:
                try:
                    del wrapper.model
                    wrapper.model = None
                    freed += 1
                except Exception:
                    pass
            if wrapper.processor is not None:
                try:
                    del wrapper.processor
                    wrapper.processor = None
                except Exception:
                    pass
            wrapper._is_loaded = False
        # 不清空 _instances，保留实例引用方便 reload
        if freed > 0:
            print(f"[木叶·显存释放] Caption模型已卸载 {freed} 个实例")
        return freed


# ────────────────────────────────────────────────────────────
def get_available_models():
    models = {}
    search_dirs = []
    try:
        import folder_paths
        base_dirs = folder_paths.get_folder_paths("checkpoints")
        if base_dirs:
            for bd in base_dirs:
                d = os.path.normpath(os.path.join(bd, "..", "Caption_checkpoints"))
                if os.path.isdir(d):
                    search_dirs.append(d)
                d2 = os.path.normpath(os.path.join(bd, "..", "LLavacheckpoints"))
                if os.path.isdir(d2):
                    search_dirs.append(d2)
        plugin_dir = os.path.dirname(os.path.dirname(__file__))
        fixed_base = os.path.normpath(os.path.join(plugin_dir, "..", "models"))
        for subdir in ["Caption_checkpoints", "LLavacheckpoints"]:
            d = os.path.normpath(os.path.join(fixed_base, subdir))
            if os.path.isdir(d) and d not in search_dirs:
                search_dirs.append(d)
        for search_dir in search_dirs:
            for item in os.listdir(search_dir):
                item_path = os.path.normpath(os.path.join(search_dir, item))
                if os.path.isdir(item_path):
                    has_model = (
                        os.path.exists(os.path.join(item_path, "config.json"))
                        or any(f.endswith(".safetensors") for f in os.listdir(item_path))
                    )
                    if has_model and item not in models:
                        models[item] = item_path
    except Exception as e:
        print(f"[木叶·图像反推] 扫描模型目录出错: {e}")
    return models


# ────────────────────────────────────────────────────────────
def detect_model_arch(model_dir):
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return "unknown"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        archs = config_dict.get("architectures", [])
        arch_str = " ".join(archs).lower()
        if "qwen2_5_vl" in arch_str or "qwen2.5_vl" in arch_str:
            return "qwen2_5_vl"
        elif "qwen2_vl" in arch_str or "qwen2.vl" in arch_str:
            return "qwen2_5_vl"
        elif "llava" in arch_str:
            return "llava"
        elif "qwen3vl" in arch_str or "qwen3_vl" in arch_str:
            return "qwen3_vl"
        model_type = (config_dict.get("model_type") or "").lower()
        if "qwen2_5_vl" in model_type or "qwen2.5_vl" in model_type:
            return "qwen2_5_vl"
        elif "qwen2_vl" in model_type or "qwen2.vl" in model_type:
            return "qwen2_5_vl"
        elif "llava" in model_type:
            return "llava"
        elif "qwen3vl" in model_type or "qwen3_vl" in model_type:
            return "qwen3_vl"
        return "unknown"
    except Exception as e:
        print(f"[木叶·图像反推] 检测架构失败: {e}")
        return "unknown"


# ────────────────────────────────────────────────────────────
def _get_quant_config(quant_mode):
    if quant_mode == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    elif quant_mode == "int4":
        return BitsAndBytesConfig(load_in_4bit=True)
    return None


# ────────────────────────────────────────────────────────────
GPU_OPTIONS = ["自动", "显卡1", "显卡2"]


def resolve_gpu_device(gpu_choice="显卡1"):
    """把节点上的显卡选项转成 from_pretrained 的 device_map 参数。

      自动  -> "auto"（accelerate 跨卡自动分配，放不下会 offload）
      显卡N -> N-1（整模型放到该卡；显卡1=cuda:0，显卡2=cuda:1）
    返回 (device_map 参数, 实际显卡索引 或 None)。
    """
    if gpu_choice == "自动":
        return "auto", None
    if not torch.cuda.is_available():
        print(f"[木叶·图像反推] 警告: CUDA 不可用，{gpu_choice} 回退为 自动")
        return "auto", None
    n = torch.cuda.device_count()
    try:
        idx = int(str(gpu_choice).replace("显卡", "")) - 1
    except ValueError:
        idx = 0
    if idx > n - 1:
        print(f"[木叶·图像反推] 警告: {gpu_choice} 不存在（共 {n} 块显卡），回退到 显卡{n}")
        idx = n - 1
    return idx, idx


# ────────────────────────────────────────────────────────────
def load_model_with_options(model_dir, quantization_mode="bf16", attention="eager",
                            gpu_choice="显卡1"):
    device_map_arg, gpu_idx = resolve_gpu_device(gpu_choice)
    if gpu_idx is not None:
        gpu_desc = f"{gpu_choice} (cuda:{gpu_idx} {torch.cuda.get_device_name(gpu_idx)})"
    else:
        gpu_desc = "自动 (auto，accelerate 跨卡分配)"
    print(f"[木叶·图像反推] 正在加载模型: {model_dir}")
    print(f"[木叶·图像反推] 量化: {quantization_mode} | 加速: {attention} | 显卡: {gpu_desc}")

    arch_type = detect_model_arch(model_dir)
    print(f"[木叶·图像反推] 模型架构: {arch_type}")

    if quantization_mode in ("int8", "int4"):
        dtype = torch.float16
    elif quantization_mode == "float16":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    quant_config = _get_quant_config(quantization_mode)
    if quant_config is not None:
        print(f"[木叶·图像反推] 使用 BitsAndBytes {quantization_mode.upper()} 量化加载")

    load_kwargs = {
        "device_map": device_map_arg,
        "dtype": dtype,
        "trust_remote_code": True,
    }
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config

    if attention != "eager":
        load_kwargs["attn_implementation"] = attention

    model = None
    processor = None

    try:
        model = _load_model_by_arch(model_dir, arch_type, load_kwargs.copy())
        model.eval()
    except Exception as e:
        print(f"[木叶·图像反推] 加载失败，回退到 eager: {e}")
        load_kwargs.pop("attn_implementation", None)
        try:
            model = _load_model_by_arch(model_dir, arch_type, load_kwargs.copy())
            model.eval()
        except Exception as e2:
            if gpu_idx is not None:
                print(f"[木叶·图像反推] {gpu_choice} 加载失败（{e2}），回退到 自动 跨卡分配重试")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                load_kwargs["device_map"] = "auto"
                gpu_idx = None
                try:
                    model = _load_model_by_arch(model_dir, arch_type, load_kwargs.copy())
                    model.eval()
                except Exception as e3:
                    print(f"[木叶·图像反推] 模型加载失败: {e3}")
                    return None, None, arch_type
            else:
                print(f"[木叶·图像反推] 模型加载失败: {e2}")
                return None, None, arch_type

    try:
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    except Exception:
        try:
            processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
        except Exception:
            processor = None

    if processor and arch_type in ("qwen2_5_vl", "qwen3_vl"):
        try:
            processor = AutoProcessor.from_pretrained(
                model_dir, max_pixels=501760, min_pixels=3136, trust_remote_code=True,
            )
        except Exception:
            pass

    if torch.cuda.is_available():
        if gpu_idx is not None:
            used = torch.cuda.memory_allocated(gpu_idx) / 1024**3
            print(f"[木叶·图像反推] 模型加载完成 | cuda:{gpu_idx} 显存占用: {used:.1f}GB")
        else:
            for i in range(torch.cuda.device_count()):
                u = torch.cuda.memory_allocated(i) / 1024**3
                if u > 0:
                    print(f"[木叶·图像反推] 模型加载完成 | cuda:{i} 显存占用: {u:.1f}GB")

    return model, processor, arch_type


def _load_model_by_arch(model_dir, arch_type, load_kwargs):
    # transformers 5.x renamed AutoModelForVision2Seq → AutoModelForImageTextToText
    try:
        from transformers import AutoModelForVision2Seq as VisionAutoModel
    except ImportError:
        try:
            from transformers import AutoModelForImageTextToText as VisionAutoModel
        except ImportError:
            VisionAutoModel = None

    if arch_type == "qwen2_5_vl":
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            return Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir, **load_kwargs)
        except (ImportError, Exception) as e:
            print(f"[木叶·图像反推] Qwen2_5_VL 专用加载失败: {e}")
    elif arch_type == "llava":
        try:
            from transformers import LlavaForConditionalGeneration
            return LlavaForConditionalGeneration.from_pretrained(model_dir, **load_kwargs)
        except (ImportError, Exception) as e:
            print(f"[木叶·图像反推] LLaVA 专用加载失败: {e}")
    elif arch_type == "qwen3_vl":
        # Qwen3-VL 系列
        try:
            from transformers import Qwen3VLForConditionalGeneration
            return Qwen3VLForConditionalGeneration.from_pretrained(model_dir, **load_kwargs)
        except (ImportError, AttributeError):
            print("[木叶·图像反推] transformers 缺少 Qwen3VLForConditionalGeneration")
        except Exception as e:
            print(f"[木叶·图像反推] Qwen3-VL 专用加载失败: {e}")
    # fallback
    if VisionAutoModel is not None:
        return VisionAutoModel.from_pretrained(model_dir, **load_kwargs)
    else:
        # Fallback to generic AutoModel
        from transformers import AutoModel
        return AutoModel.from_pretrained(model_dir, **load_kwargs)


# ════════════════════════════════════════════════════════════
class 反推模型加载:
    @classmethod
    def INPUT_TYPES(cls):
        model_dirs = get_available_models()
        return {
            "required": {
                "模型选择": (list(model_dirs.keys()) if model_dirs else ["(无可用模型)"], {
                    "default": list(model_dirs.keys())[0] if model_dirs else "(无可用模型)",
                    "tooltip": "选择使用的反推模型",
                }),
                "量化方式": (
                    ["BF16", "FP16", "INT8", "INT4"],
                    {"default": "BF16", "tooltip": "模型量化方式"},
                ),
                "显卡选择": (
                    GPU_OPTIONS,
                    {"default": "显卡1", "tooltip": "显卡1=cuda:0，显卡2=cuda:1，自动=accelerate 跨卡分配。指定显卡显存不足会自动回退为自动分配"},
                ),
                "attention": (
                    ["eager", "sdpa", "flash_attention_2"],
                    {"default": "eager", "tooltip": "注意力加速方式"},
                ),
            },
        }

    RETURN_TYPES = (CAPTION_MODEL,)
    RETURN_NAMES = ("模型",)
    FUNCTION = "load_model"
    CATEGORY = "Muye/图像反推"

    def load_model(self, 模型选择, 量化方式="BF16", attention="eager", 显卡选择="显卡1"):
        model_dirs = get_available_models()
        model_path = model_dirs.get(模型选择)

        if not model_path or not os.path.isdir(model_path):
            print(f"[木叶·图像反推] 警告: 找不到模型 '{模型选择}'")
            return (CaptionModelWrapper(None, None, "unknown", "", 模型选择,
                                        gpu_choice=显卡选择),)

        quant_mode_map = {
            "BF16": "bf16", "FP16": "float16",
            "INT8": "int8", "INT4": "int4",
        }

        quant_str = quant_mode_map.get(量化方式, "bf16")
        model, processor, arch_type = load_model_with_options(
            model_path,
            quant_str,
            attention,
            显卡选择,
        )

        if model is None:
            return (CaptionModelWrapper(None, None, "unknown", "", 模型选择,
                                        gpu_choice=显卡选择),)

        wrapper = CaptionModelWrapper(
            model=model, processor=processor, arch_type=arch_type,
            model_path=model_path, model_name=模型选择,
            quantization=quant_str, attention=attention,
            gpu_choice=显卡选择,
        )
        print(f"[木叶·图像反推] 模型已加载: {wrapper}")
        return (wrapper,)


NODE_CLASS_MAPPINGS = {
    "反推模型加载": 反推模型加载,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "反推模型加载": "🦞 反推模型加载",
}
