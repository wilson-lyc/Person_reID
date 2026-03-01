import argparse
import os
import traceback

import timm


TIMM_BACKBONES = {
    "swin": "swin_base_patch4_window7_224",
    "swinv2": "swinv2_base_window8_256",
    "dino": "vit_base_patch16_dinov3.lvd1689m",
    "convnext": "convnext_base",
    "hrnet": "hrnet_w18",
}

def _set_cache_env(cache_dir: str | None) -> None:
    if not cache_dir:
        return
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = os.path.join(cache_dir, "hf")
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(cache_dir, "hf", "hub")
    os.environ["TORCH_HOME"] = os.path.join(cache_dir, "torch")
    os.environ["TIMM_CACHE_DIR"] = os.path.join(cache_dir, "timm")
    print(f"[cache] root={cache_dir}")
    print(f"[cache] HF_HOME={os.environ['HF_HOME']}")
    print(f"[cache] TORCH_HOME={os.environ['TORCH_HOME']}")
    print(f"[cache] TIMM_CACHE_DIR={os.environ['TIMM_CACHE_DIR']}")


def _set_hf_endpoint(use_hf_mirror: bool) -> None:
    if use_hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(f"[hf] HF_ENDPOINT={os.environ['HF_ENDPOINT']}")


def preload_timm_model(model_name: str) -> None:
    print(f"[timm] downloading pretrained weights: {model_name}")
    model = timm.create_model(model_name, pretrained=True)
    del model


def preload_hf_repo(repo_id: str, cache_dir: str | None) -> None:
    print(f"[hf] snapshot_download: {repo_id}")
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, cache_dir=cache_dir)


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
        "--hf-repos",
        type=str,
        default="",
        help="extra Hugging Face repo ids to snapshot_download, comma-separated",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="",
        help="custom cache root. Will set HF_HOME/TORCH_HOME/TIMM_CACHE_DIR under it.",
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
    cache_dir = args.cache_dir.strip() or None
    _set_cache_env(cache_dir)
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
            "Please pass at least one flag, e.g. --use_swin or "
            "--use_resnet50 --use_dense"
        )
        return 2

    print(f"[run] models={targets}")
    if args.hf_repos.strip():
        print(f"[run] extra hf repos={args.hf_repos}")

    failed: list[str] = []

    for target in targets:
        try:
            if target in TIMM_BACKBONES:
                preload_timm_model(TIMM_BACKBONES[target])
            else:
                print(f"[skip] unknown target: {target}")
        except Exception:
            failed.append(target)
            print(f"[error] failed target: {target}")
            print(traceback.format_exc())

    extra_repos = [x.strip() for x in args.hf_repos.split(",") if x.strip()]
    for repo_id in extra_repos:
        try:
            preload_hf_repo(repo_id, cache_dir)
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
