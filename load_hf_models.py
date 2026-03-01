import argparse
import os
import traceback

import timm
import torch
import torch.nn as nn


def _force_cpu_only() -> None:
    # Ensure this script never depends on CUDA runtime.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    try:
        torch.set_default_device("cpu")
    except Exception:
        # Keep compatibility with older torch versions.
        pass
    print("[device] cpu-only mode enabled")


def _set_hf_endpoint(use_hf_mirror: bool) -> None:
    if use_hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(f"[hf] HF_ENDPOINT={os.environ['HF_ENDPOINT']}")


def preload_swin() -> None:
    print("[timm] preload path: swin_base_patch4_window7_224 (pretrained=True, drop_path_rate=0.2)")
    model_ft = timm.create_model(
        "swin_base_patch4_window7_224", pretrained=True, drop_path_rate=0.2
    )
    model_ft = model_ft.to("cpu")
    # Align with model.py memory-optimized head replacement.
    model_ft.head = nn.Sequential()
    del model_ft


def preload_swinv2(input_size: tuple[int, int]) -> None:
    print(
        "[timm] preload path: swinv2_base_window8_256 "
        f"(pretrained=False, img_size={input_size}, drop_path_rate=0.2) + load pretrained state"
    )
    model_ft = timm.create_model(
        "swinv2_base_window8_256",
        pretrained=False,
        img_size=input_size,
        drop_path_rate=0.2,
    )
    model_ft = model_ft.to("cpu")
    model_full = timm.create_model("swinv2_base_window8_256", pretrained=True)
    model_full = model_full.to("cpu")
    model_ft.load_state_dict(model_full.state_dict(), strict=False)
    model_ft.head = nn.Sequential()
    del model_ft
    del model_full


def preload_dino(input_size: tuple[int, int]) -> None:
    print(
        "[timm] preload path: vit_base_patch16_dinov3.lvd1689m "
        f"(pretrained=False, img_size={input_size}, drop_path_rate=0.2) + load pretrained state"
    )
    model_ft = timm.create_model(
        "vit_base_patch16_dinov3.lvd1689m",
        pretrained=False,
        img_size=input_size,
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
    print("[timm] preload path: convnext_base (pretrained=True, drop_path_rate=0.2)")
    model_ft = timm.create_model("convnext_base", pretrained=True, drop_path_rate=0.2)
    model_ft = model_ft.to("cpu")
    model_ft.head = nn.Sequential()
    del model_ft


def preload_hrnet() -> None:
    print("[timm] preload path: hrnet_w18 (pretrained=True)")
    model_ft = timm.create_model("hrnet_w18", pretrained=True)
    model_ft = model_ft.to("cpu")
    model_ft.classifier = nn.Sequential()
    del model_ft


def preload_hf_repo(repo_id: str) -> None:
    print(f"[hf] snapshot_download: {repo_id}")
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preload/download model files locally (no training)."
    )
    parser.add_argument("--use_swin", action="store_true", help="preload Swin-Base")
    parser.add_argument("--use_swinv2", action="store_true", help="preload SwinV2-Base")
    parser.add_argument("--use_dino", action="store_true", help="preload DINOv3 ViT-Base")
    parser.add_argument("--use_convnext", action="store_true", help="preload ConvNeXt-Base")
    parser.add_argument("--use_hr", action="store_true", help="preload HRNet-W18")
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="input height for swinv2/dino preload path to match training model construction",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="input width for swinv2/dino preload path to match training model construction",
    )
    parser.add_argument(
        "--hf-repos",
        type=str,
        default="",
        help="extra Hugging Face repo ids to snapshot_download, comma-separated",
    )
    parser.add_argument(
        "--use-hf-mirror",
        "--use_hf_mirror",
        action="store_true",
        help="use Hugging Face mirror by setting HF_ENDPOINT=https://hf-mirror.com",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _force_cpu_only()
    _set_hf_endpoint(args.use_hf_mirror)

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
        print("[error] no backbone selected.")
        print(
            "Please pass at least one flag, e.g. --use_swin or --use_dino"
        )
        return 2

    input_size = (args.height, args.width)
    print(f"[run] models={targets}")
    print(f"[run] input_size={input_size}")
    if args.hf_repos.strip():
        print(f"[run] extra hf repos={args.hf_repos}")

    failed: list[str] = []

    for target in targets:
        try:
            if target == "swin":
                preload_swin()
            elif target == "swinv2":
                preload_swinv2(input_size)
            elif target == "dino":
                preload_dino(input_size)
            elif target == "convnext":
                preload_convnext()
            elif target == "hrnet":
                preload_hrnet()
        except Exception:
            failed.append(target)
            print(f"[error] failed target: {target}")
            print(traceback.format_exc())

    extra_repos = [x.strip() for x in args.hf_repos.split(",") if x.strip()]
    for repo_id in extra_repos:
        try:
            preload_hf_repo(repo_id)
        except Exception:
            failed.append(f"hf:{repo_id}")
            print(f"[error] failed hf repo: {repo_id}")
            print(traceback.format_exc())

    if failed:
        print(f"[done] finished with failures: {failed}")
        return 1

    print("[done] all requested preloads succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
