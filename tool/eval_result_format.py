import os
from datetime import datetime


DATASET_NAME_MAP = {
    "market": "Market-1501",
    "duke": "DukeMTMC-reID",
    "msmt": "MSMT17",
    "cub": "CUB-200-2011",
    "vehicleid": "VehicleID",
    "veri": "VeRi",
    "viper": "VIPeR",
}


def infer_model_name(result_mat_path):
    model_name = os.path.basename(os.path.dirname(os.path.abspath(result_mat_path)))
    return model_name or "unknown_model"


def infer_eval_dataset(result_mat_path):
    basename = os.path.basename(result_mat_path)
    stem = os.path.splitext(basename)[0].lower()
    prefix = "pytorch_result_"
    if stem.startswith(prefix):
        dataset_tag = stem[len(prefix):]
        return DATASET_NAME_MAP.get(dataset_tag, "Unknown")
    return "Unknown"


def format_elapsed(seconds):
    minutes = int(seconds // 60)
    remain_seconds = seconds % 60
    return f"Evaluation complete in {minutes}m {remain_seconds:.2f}s"


def format_torch_size(shape):
    return f"torch.Size([{', '.join(str(x) for x in shape)}])"


def build_result_block(
    *,
    evaluated_at,
    model_name,
    eval_dataset,
    result_mat,
    elapsed_text,
    rank1,
    rank5,
    rank10,
    map_score,
    feature_shape_text=None,
    extra_metric_lines=None,
):
    lines = [
        "================== Evaluation Result ===================",
        f"evaluated_at    : {evaluated_at}",
        f"model_name      : {model_name}",
        f"eval_dataset    : {eval_dataset}",
        f"result_mat      : {result_mat}",
        f"elapsed         : {elapsed_text}",
        "========================================================",
    ]
    if feature_shape_text:
        lines.append(feature_shape_text)
    lines.append("--------------------------------------------------------")
    lines.append(
        f"Rank@1:{rank1:.6f} Rank@5:{rank5:.6f} Rank@10:{rank10:.6f} mAP:{map_score:.6f}"
    )
    if extra_metric_lines:
        lines.extend(extra_metric_lines)
    lines.append("========================================================")
    return "\n".join(lines)


def append_result_txt(result_mat_path, block):
    result_txt_path = os.path.join(os.path.dirname(os.path.abspath(result_mat_path)), "result.txt")
    separator = ""
    if os.path.isfile(result_txt_path) and os.path.getsize(result_txt_path) > 0:
        with open(result_txt_path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            last_char = f.read(1)
        separator = "\n" if last_char == b"\n" else "\n\n"
    with open(result_txt_path, "a", encoding="utf-8") as f:
        f.write(separator)
        f.write(block.rstrip("\n"))
        f.write("\n")
    return result_txt_path


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
