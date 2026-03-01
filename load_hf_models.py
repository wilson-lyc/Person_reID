import argparse
import os
import sys
import traceback

import timm
import torch
import torch.nn as nn

MODEL_INPUT_SIZE = (256, 128)
_ENABLE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None
_COLOR_RESET = "\033[0m"
_COLOR_MAP = {
    "success": "\033[1;32m",
    "fail": "\033[1;31m",
    "info": "\033[1;36m",
}


def _log(tag: str, message: str) -> None:
    label = f"[{tag}]"
    if _ENABLE_COLOR:
        color = _COLOR_MAP.get(tag, "")
        if color:
            label = f"{color}{label}{_COLOR_RESET}"
    print(f"{label} {message}")


def _force_cpu_only() -> None:
    # Ensure this script never depends on CUDA runtime.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    try:
        torch.set_default_device("cpu")
    except Exception:
        # Keep compatibility with older torch versions.
        pass


def preload_swin() -> None:
    model_ft = timm.create_model(
        "swin_base_patch4_window7_224", pretrained=True, drop_path_rate=0.2
    )
    model_ft = model_ft.to("cpu")
    # Align with model.py memory-optimized head replacement.
    model_ft.head = nn.Sequential()
    del model_ft


def preload_swinv2() -> None:
    model_ft = timm.create_model(
        "swinv2_base_window8_256",
        pretrained=False,
        img_size=MODEL_INPUT_SIZE,
        drop_path_rate=0.2,
    )
    model_ft = model_ft.to("cpu")
    model_full = timm.create_model("swinv2_base_window8_256", pretrained=True)
    model_full = model_full.to("cpu")
    model_ft.load_state_dict(model_full.state_dict(), strict=False)
    model_ft.head = nn.Sequential()
    del model_ft
    del model_full


def preload_dino() -> None:
    model_ft = timm.create_model(
        "vit_base_patch16_dinov3.lvd1689m",
        pretrained=False,
        img_size=MODEL_INPUT_SIZE,
        drop_path_rate=0.2,
    )
    model_ft = model_ft.to("cpu")
    model_full = timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=True)
    model_full = model_full.to("cpu")
    model_ft.load_state_dict(model_full.state_dict(), strict=False)
    model_ft.head = nn.Sequential()
    del model_ft
    del model_full


def preload_convnext() -> None:
    model_ft = timm.create_model("convnext_base", pretrained=True, drop_path_rate=0.2)
    model_ft = model_ft.to("cpu")
    model_ft.head = nn.Sequential()
    del model_ft


def preload_hrnet() -> None:
    model_ft = timm.create_model("hrnet_w18", pretrained=True)
    model_ft = model_ft.to("cpu")
    model_ft.classifier = nn.Sequential()
    del model_ft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preload/download model files locally (no training)."
    )
    parser.add_argument("--use_swin", action="store_true", help="preload Swin-Base")
    parser.add_argument("--use_swinv2", action="store_true", help="preload SwinV2-Base")
    parser.add_argument("--use_dino", action="store_true", help="preload DINOv3 ViT-Base")
    parser.add_argument("--use_convnext", action="store_true", help="preload ConvNeXt-Base")
    parser.add_argument("--use_hr", action="store_true", help="preload HRNet-W18")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _force_cpu_only()

    targets: list[str] = []
    if args.use_swin:
        targets.append("swin")
    if args.use_swinv2:
        targets.append("swinv2")
    if args.use_dino:
        targets.append("dino")
    if args.use_convnext:
        targets.append("convnext")
    if args.use_hr:
        targets.append("hrnet")

    if not targets:
        _log("fail", "no backbone selected.")
        _log("info", "Please pass at least one flag, e.g. --use_swin or --use_dino")
        return 2

    failed: list[str] = []

    for target in targets:
        try:
            if target == "swin":
                preload_swin()
            elif target == "swinv2":
                preload_swinv2()
            elif target == "dino":
                preload_dino()
            elif target == "convnext":
                preload_convnext()
            elif target == "hrnet":
                preload_hrnet()
        except Exception:
            failed.append(target)
            _log("fail", f"failed target: {target}")
            print(traceback.format_exc())

    if failed:
        _log("fail", f"finished with failures: {failed}")
        return 1

    _log("success", "all requested preloads succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
