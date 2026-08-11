import torch
import gc
import sys
import comfy.model_management as model_management


# ────────────────────────────────────────────────────────────
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")


def _unload_caption_models():
    """从 sys.modules 中找到 CaptionModelWrapper 并卸载"""
    mod_name = "image2prompt.model_loader"
    if mod_name not in sys.modules:
        # 尝试找所有可能加载了的 model_loader 模块
        found = False
        for key, mod in list(sys.modules.items()):
            if hasattr(mod, "CaptionModelWrapper"):
                wrapper_cls = getattr(mod, "CaptionModelWrapper", None)
                if wrapper_cls is not None and hasattr(wrapper_cls, "_instances"):
                    freed = wrapper_cls.unload_all()
                    found = True
                    print(f"[木叶·显存释放] 通过 {key} 找到并卸载 Caption 模型")
                    return freed
        if not found:
            print("[木叶·显存释放] 未找到 model_loader 模块，尝试直接扫描 GPU 引用...")
        return 0

    mod = sys.modules[mod_name]
    wrapper_cls = getattr(mod, "CaptionModelWrapper", None)
    if wrapper_cls is None or not hasattr(wrapper_cls, "_instances"):
        print("[木叶·显存释放] CaptionModelWrapper 不符合预期")
        return 0

    freed = wrapper_cls.unload_all()
    return freed


class 显存释放:
    """
    木叶节点 - 显存释放
    同时清理 ComfyUI 管理的模型和木叶自定义模型（CaptionModel）
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (any, ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True

    RETURN_TYPES = (any, )
    FUNCTION = "cleanup"
    CATEGORY = "Muye/工具"

    def cleanup(self, **kwargs):
        print("[木叶·显存释放] 开始清理显存...")

        # 1. 先卸载木叶 Caption 模型（在 ComfyUI unload 之前，因为后面会清缓存）
        freed = _unload_caption_models()

        # 3. 卸载 ComfyUI 管理的所有模型
        try:
            model_management.unload_all_models()
            model_management.soft_empty_cache(True)
            print("[木叶·显存释放] ComfyUI 模型已卸载")
        except Exception as e:
            print(f"[木叶·显存释放] ComfyUI 模型卸载异常(可忽略): {e}")

        # 4. Caption 模型的 tensor 引用已断开，再清一轮
        gc.collect()
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass

        # 5. 再试一次卸载 ComfyUI 模型（有时需要两轮才能彻底清干净）
        try:
            model_management.unload_all_models()
            model_management.soft_empty_cache(True)
        except Exception:
            pass
        gc.collect()
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass

        # 6. 汇报
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated(0) / 1024**3
            print(f"[木叶·显存释放] 清理完成 | 当前显存占用: {used:.1f}GB")

        return (kwargs.get("value"),)


NODE_CLASS_MAPPINGS = {
    "显存释放": 显存释放,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "显存释放": "🦞 显存释放",
}
