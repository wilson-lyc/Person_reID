import argparse
import os
from shutil import copyfile


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def iter_jpg_files(folder: str):
    for _, _, files in os.walk(folder, topdown=True):
        for name in files:
            if name.lower().endswith(".jpg"):
                yield name


def prepare_market_or_duke(download_path: str) -> None:
    save_path = os.path.join(download_path, "pytorch")
    ensure_dir(save_path)

    # query
    query_path = os.path.join(download_path, "query")
    query_save_path = os.path.join(save_path, "query")
    ensure_dir(query_save_path)
    for name in iter_jpg_files(query_path):
        person_id = name.split("_")[0]
        src_path = os.path.join(query_path, name)
        dst_dir = os.path.join(query_save_path, person_id)
        ensure_dir(dst_dir)
        copyfile(src_path, os.path.join(dst_dir, name))

    # multi-query (optional)
    multi_query_path = os.path.join(download_path, "gt_bbox")
    if os.path.isdir(multi_query_path):
        multi_query_save_path = os.path.join(save_path, "multi-query")
        ensure_dir(multi_query_save_path)
        for name in iter_jpg_files(multi_query_path):
            person_id = name.split("_")[0]
            src_path = os.path.join(multi_query_path, name)
            dst_dir = os.path.join(multi_query_save_path, person_id)
            ensure_dir(dst_dir)
            copyfile(src_path, os.path.join(dst_dir, name))

    # gallery
    gallery_path = os.path.join(download_path, "bounding_box_test")
    gallery_save_path = os.path.join(save_path, "gallery")
    ensure_dir(gallery_save_path)
    for name in iter_jpg_files(gallery_path):
        person_id = name.split("_")[0]
        src_path = os.path.join(gallery_path, name)
        dst_dir = os.path.join(gallery_save_path, person_id)
        ensure_dir(dst_dir)
        copyfile(src_path, os.path.join(dst_dir, name))

    # train_all
    train_path = os.path.join(download_path, "bounding_box_train")
    train_all_save_path = os.path.join(save_path, "train_all")
    ensure_dir(train_all_save_path)
    for name in iter_jpg_files(train_path):
        person_id = name.split("_")[0]
        src_path = os.path.join(train_path, name)
        dst_dir = os.path.join(train_all_save_path, person_id)
        ensure_dir(dst_dir)
        copyfile(src_path, os.path.join(dst_dir, name))

    # train / val split
    train_save_path = os.path.join(save_path, "train")
    val_save_path = os.path.join(save_path, "val")
    ensure_dir(train_save_path)
    ensure_dir(val_save_path)
    for name in iter_jpg_files(train_path):
        person_id = name.split("_")[0]
        src_path = os.path.join(train_path, name)
        train_person_dir = os.path.join(train_save_path, person_id)
        val_person_dir = os.path.join(val_save_path, person_id)
        if not os.path.isdir(train_person_dir):
            ensure_dir(train_person_dir)
            ensure_dir(val_person_dir)  # first image goes to val
            dst_dir = val_person_dir
        else:
            dst_dir = train_person_dir
        copyfile(src_path, os.path.join(dst_dir, name))


def prepare_msmt17(download_path: str) -> None:
    save_path = os.path.join(download_path, "pytorch")
    ensure_dir(save_path)

    query_save_path = os.path.join(save_path, "query")
    gallery_save_path = os.path.join(save_path, "gallery")
    train_save_path = os.path.join(save_path, "train")
    train_all_save_path = os.path.join(save_path, "train_all")
    val_save_path = os.path.join(save_path, "val")

    for p in [query_save_path, gallery_save_path, train_save_path, train_all_save_path, val_save_path]:
        ensure_dir(p)

    test_root = os.path.join(download_path, "test")
    train_root = os.path.join(download_path, "train")

    with open(os.path.join(download_path, "list_query.txt"), "r") as f:
        for line in f:
            rel = line.split(" ")[0].strip()
            person_id = rel.split("/")[0]
            src_path = os.path.join(test_root, rel)
            dst_dir = os.path.join(query_save_path, person_id)
            ensure_dir(dst_dir)
            copyfile(src_path, os.path.join(dst_dir, os.path.basename(rel)))

    with open(os.path.join(download_path, "list_gallery.txt"), "r") as f:
        for line in f:
            rel = line.split(" ")[0].strip()
            person_id = rel.split("/")[0]
            src_path = os.path.join(test_root, rel)
            dst_dir = os.path.join(gallery_save_path, person_id)
            ensure_dir(dst_dir)
            copyfile(src_path, os.path.join(dst_dir, os.path.basename(rel)))

    with open(os.path.join(download_path, "list_train.txt"), "r") as f:
        for line in f:
            rel = line.split(" ")[0].strip()
            person_id = rel.split("/")[0]
            src_path = os.path.join(train_root, rel)
            dst_train = os.path.join(train_save_path, person_id)
            dst_all = os.path.join(train_all_save_path, person_id)
            ensure_dir(dst_train)
            ensure_dir(dst_all)
            filename = os.path.basename(rel)
            copyfile(src_path, os.path.join(dst_train, filename))
            copyfile(src_path, os.path.join(dst_all, filename))

    with open(os.path.join(download_path, "list_val.txt"), "r") as f:
        for line in f:
            rel = line.split(" ")[0].strip()
            person_id = rel.split("/")[0]
            src_path = os.path.join(train_root, rel)
            dst_val = os.path.join(val_save_path, person_id)
            dst_all = os.path.join(train_all_save_path, person_id)
            ensure_dir(dst_val)
            ensure_dir(dst_all)
            filename = os.path.basename(rel)
            copyfile(src_path, os.path.join(dst_val, filename))
            copyfile(src_path, os.path.join(dst_all, filename))


def resolve_download_path(dataset: str, requested: str) -> str:
    if requested:
        return requested
    defaults = {
        "market": "./Market",
        "duke": "./DukeMTMC-reID",
        "msmt17": "./MSMT17_V1",
    }
    return defaults[dataset]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare re-ID dataset to torchvision ImageFolder format")
    parser.add_argument("--dataset", default="market", choices=["market", "duke", "msmt17"], help="dataset type")
    parser.add_argument("--download_path", default="", type=str, help="path to raw downloaded dataset root")
    parser.add_argument(
        "--market_legacy_path",
        default="./Market-1501-v15.09.15",
        type=str,
        help="legacy Market-1501 folder path to rename when --dataset market",
    )
    opt = parser.parse_args()

    download_path = resolve_download_path(opt.dataset, opt.download_path)

    if opt.dataset == "market" and not os.path.isdir(download_path) and os.path.isdir(opt.market_legacy_path):
        os.rename(opt.market_legacy_path, download_path)

    if not os.path.isdir(download_path):
        raise FileNotFoundError(f"Dataset path does not exist: {download_path}")

    if opt.dataset in ("market", "duke"):
        prepare_market_or_duke(download_path)
    else:
        prepare_msmt17(download_path)

    print(f"Prepared dataset '{opt.dataset}' at: {os.path.join(download_path, 'pytorch')}")


if __name__ == "__main__":
    main()
