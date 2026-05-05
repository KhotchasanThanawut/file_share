#!/usr/bin/env python3
"""
Generate TPU-MLIR model_transform and model_deploy commands
for PaddleOCR DET and REC ONNX models.

This script only PRINTS commands.
It does NOT run model_transform or model_deploy.
"""

from pathlib import Path
import ast
import sys

import yaml
import onnx
from onnx import TensorProto


# ============================================================
# USER CONFIG AREA
# ============================================================

# Interactive mode:
#   None  = ask from menu when program starts
#   "det" = directly generate DET
#   "rec" = directly generate REC
#   "both" = generate both
TARGET_MODEL = None


# ------------------------------------------------------------
# 4 manual paths
# ------------------------------------------------------------

DET_CONFIG_PATH = "/home/kdt/intern/model_convert/models/indo_model_22-04-2026/det_rec_model/config_det_indo.yml"
DET_ONNX_PATH   = "/home/kdt/intern/model_convert/models/indo_model_22-04-2026/det_rec_model/INDO_det_model_20260429.onnx"

REC_CONFIG_PATH = "/home/kdt/intern/model_convert/models/indo_model_22-04-2026/det_rec_model/config_rec_indo.yml"
REC_ONNX_PATH   = "/home/kdt/intern/model_convert/models/indo_model_22-04-2026/det_rec_model/indo_rec_ppocr_no_chars_v4_setSpaceChar_false_20260424.onnx"

# ------------------------------------------------------------
# model_transform config
# ------------------------------------------------------------

DET_MODEL_NAME = "ppocr_det"
REC_MODEL_NAME = "ppocr_rec"

DET_MLIR_PATH = "ppocr_det.mlir"
REC_MLIR_PATH = "ppocr_rec.mlir"

DET_TEST_INPUT = "../image/det_test.jpg"
REC_TEST_INPUT = "../image/rec_test.jpg"

DET_TEST_RESULT = "ppocr_det_top_outputs.npz"
REC_TEST_RESULT = "ppocr_rec_top_outputs.npz"

# DET usually needs manual fixed shape.
# Example: [1, 3, 640, 640]
DET_INPUT_SHAPE = [1, 3, 640, 640]

# REC can usually be read from config_rec.yml:
#   RecResizeImg.image_shape: [3, 48, 320]
#
# Set manually if needed:
#   REC_INPUT_SHAPE = [1, 3, 48, 320]
REC_INPUT_SHAPE = None

# Usually False.
# Enable only if you intentionally want TPU-MLIR image preprocessing
# to keep image aspect ratio.
DET_KEEP_ASPECT_RATIO = False
REC_KEEP_ASPECT_RATIO = False

# Usually None.
# Set only if TPU-MLIR requires explicit output names.
DET_OUTPUT_NAMES = None
REC_OUTPUT_NAMES = None

# Example:
# DET_OUTPUT_NAMES = ["save_infer_model/scale_0.tmp_1"]
# REC_OUTPUT_NAMES = ["softmax_0.tmp_0"]


# ------------------------------------------------------------
# model_deploy config
# ------------------------------------------------------------

PRINT_MODEL_DEPLOY = True

# For CV186AH, use "cv186x"
# For BM1688, use "bm1688"
PROCESSOR = "cv186x"

# Recommended first test: F16
# INT8 needs calibration table.
QUANTIZE = "F16"

DET_DEPLOY_TEST_INPUT = "ppocr_det_in_f32.npz"
REC_DEPLOY_TEST_INPUT = "ppocr_rec_in_f32.npz"

DET_BMODEL_PATH = "ppocr_det_cv186x_f32.bmodel"
REC_BMODEL_PATH = "ppocr_rec_cv186x_f32.bmodel"

# INT8 only.
# Leave None for F16.
DET_CALIBRATION_TABLE = None
REC_CALIBRATION_TABLE = None

# Example for INT8:
# QUANTIZE = "INT8"
# DET_CALIBRATION_TABLE = "ppocr_det_cali_table"
# REC_CALIBRATION_TABLE = "ppocr_rec_cali_table"


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_number(value):
    """
    Convert YAML number/string like:
      0.5
      "1./255."
      "1/255"
    into float safely.
    """
    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        raise ValueError(f"Cannot parse number from: {value!r}")

    text = value.strip()
    allowed_chars = set("0123456789.+-*/() ")

    if not set(text) <= allowed_chars:
        raise ValueError(f"Unsafe numeric expression in YAML: {text!r}")

    node = ast.parse(text, mode="eval")
    return float(eval(compile(node, "<safe_number>", "eval"), {"__builtins__": {}}, {}))


def fmt_float(x):
    return f"{x:.10g}"


def comma_list(values):
    return ",".join(fmt_float(x) for x in values)


def shape_to_tpu_mlir(shape):
    return "[[" + ",".join(str(x) for x in shape) + "]]"


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dtype_to_str(dtype):
    return TensorProto.DataType.Name(dtype)


# ============================================================
# ONNX HELPERS
# ============================================================

def get_value_info_shape(value_info):
    shape = []
    tensor_type = value_info.type.tensor_type

    for dim in tensor_type.shape.dim:
        if dim.dim_value > 0:
            shape.append(dim.dim_value)
        elif dim.dim_param:
            shape.append(str(dim.dim_param))
        else:
            shape.append("?")

    return shape


def is_dynamic_shape(shape):
    return any(not isinstance(x, int) for x in shape)


def get_real_model_inputs(model):
    initializer_names = {x.name for x in model.graph.initializer}
    return [x for x in model.graph.input if x.name not in initializer_names]


def get_onnx_info(onnx_path):
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    real_inputs = get_real_model_inputs(model)

    if not real_inputs:
        raise RuntimeError("No real runtime inputs found in ONNX model.")

    if len(real_inputs) > 1:
        print("WARNING: More than one real ONNX input found.", file=sys.stderr)
        print("The script will use the first input only.", file=sys.stderr)
        for inp in real_inputs:
            print(f"  input: {inp.name}", file=sys.stderr)

    input_info = real_inputs[0]
    input_tensor_type = input_info.type.tensor_type

    outputs = []
    for out in model.graph.output:
        tensor_type = out.type.tensor_type
        outputs.append({
            "name": out.name,
            "dtype": dtype_to_str(tensor_type.elem_type),
            "shape": get_value_info_shape(out),
        })

    return {
        "input_name": input_info.name,
        "input_dtype": dtype_to_str(input_tensor_type.elem_type),
        "input_shape": get_value_info_shape(input_info),
        "output_names": [x["name"] for x in outputs],
        "outputs": outputs,
    }


# ============================================================
# YAML TRANSFORM HELPERS
# ============================================================

def find_transform(config, section_name, transform_name):
    section = config.get(section_name, {})
    dataset = section.get("dataset", {})
    transforms = dataset.get("transforms", [])

    for item in transforms:
        if isinstance(item, dict) and transform_name in item:
            return item[transform_name]

    return None


def find_transform_prefer_eval(config, transform_name):
    result = find_transform(config, "Eval", transform_name)
    if result is not None:
        return result

    result = find_transform(config, "Train", transform_name)
    if result is not None:
        return result

    return None


def get_decode_img_mode(config):
    decode_cfg = find_transform_prefer_eval(config, "DecodeImage")

    if decode_cfg is None:
        raise RuntimeError("DecodeImage not found in Eval/Train transforms.")

    img_mode = decode_cfg.get("img_mode", "BGR").lower()

    if img_mode not in {"bgr", "rgb", "gray"}:
        raise RuntimeError(f"Unsupported img_mode: {img_mode}")

    return img_mode


# ============================================================
# DET PREPROCESS
# ============================================================

def get_det_preprocess(config):
    img_mode = get_decode_img_mode(config)

    norm_cfg = find_transform_prefer_eval(config, "NormalizeImage")
    to_chw_cfg = find_transform_prefer_eval(config, "ToCHWImage")

    if norm_cfg is None:
        raise RuntimeError("DET config requires NormalizeImage but it was not found.")

    norm_scale = safe_number(norm_cfg.get("scale", 1.0))
    mean = [safe_number(x) for x in norm_cfg.get("mean", [])]
    std = [safe_number(x) for x in norm_cfg.get("std", [])]

    if len(mean) != len(std):
        raise RuntimeError(f"mean/std length mismatch: mean={mean}, std={std}")

    if len(mean) not in {1, 3}:
        raise RuntimeError(f"Expected 1 or 3 channels, got mean={mean}, std={std}")

    # PaddleOCR DET NormalizeImage:
    #   y = (x * norm_scale - mean) / std
    #
    # TPU-MLIR:
    #   y = (x - tpu_mean) * tpu_scale
    #
    # Therefore:
    #   tpu_mean  = mean / norm_scale
    #   tpu_scale = norm_scale / std
    tpu_mean = [m / norm_scale for m in mean]
    tpu_scale = [norm_scale / s for s in std]

    channel_format = "nchw" if to_chw_cfg is not None else "nhwc"

    return {
        "pixel_format": img_mode,
        "channel_format": channel_format,
        "mean": tpu_mean,
        "scale": tpu_scale,
    }


# ============================================================
# REC PREPROCESS
# ============================================================

def get_rec_image_shape(config):
    rec_resize_cfg = find_transform_prefer_eval(config, "RecResizeImg")

    if rec_resize_cfg is not None:
        image_shape = rec_resize_cfg.get("image_shape")
        if image_shape:
            return [int(x) for x in image_shape]

    global_shape = config.get("Global", {}).get("d2s_train_image_shape")
    if global_shape:
        return [int(x) for x in global_shape]

    raise RuntimeError(
        "REC config requires RecResizeImg.image_shape or Global.d2s_train_image_shape."
    )


def get_rec_preprocess(config):
    img_mode = get_decode_img_mode(config)
    image_shape = get_rec_image_shape(config)

    c = int(image_shape[0])

    # PaddleOCR REC RecResizeImg:
    #   y = x / 255
    #   y = y - 0.5
    #   y = y / 0.5
    #
    # Equivalent TPU-MLIR:
    #   y = (x - 127.5) * (1 / 127.5)
    tpu_mean = [127.5] * c
    tpu_scale = [1.0 / 127.5] * c

    return {
        "pixel_format": img_mode,
        "channel_format": "nchw",
        "mean": tpu_mean,
        "scale": tpu_scale,
        "image_shape": image_shape,
    }


# ============================================================
# SHAPE RESOLUTION
# ============================================================

def resolve_input_shape(model_type, config, onnx_info, manual_shape):
    if manual_shape is not None:
        return manual_shape

    if model_type == "rec":
        c, h, w = get_rec_image_shape(config)
        return [1, c, h, w]

    onnx_shape = onnx_info["input_shape"]

    if not is_dynamic_shape(onnx_shape):
        return onnx_shape

    raise RuntimeError(
        f"{model_type.upper()} ONNX input shape is dynamic: {onnx_shape}\n"
        "Set manual input shape in USER CONFIG AREA."
    )


# ============================================================
# COMMAND PRINTERS
# ============================================================

def print_model_info(model_type, config_path, onnx_path, onnx_info, input_shape, preprocess):
    print(f"# ===== {model_type.upper()} INFO =====")
    print(f"# config         : {config_path}")
    print(f"# onnx           : {onnx_path}")
    print(f"# input name     : {onnx_info['input_name']}")
    print(f"# input dtype    : {onnx_info['input_dtype']}")
    print(f"# onnx shape     : {onnx_info['input_shape']}")
    print(f"# used shape     : {input_shape}")
    print(f"# pixel_format   : {preprocess['pixel_format']}")
    print(f"# channel_format : {preprocess['channel_format']}")
    print(f"# mean           : {comma_list(preprocess['mean'])}")
    print(f"# scale          : {comma_list(preprocess['scale'])}")

    for i, out in enumerate(onnx_info["outputs"]):
        print(f"# output {i} name : {out['name']}")
        print(f"# output {i} dtype: {out['dtype']}")
        print(f"# output {i} shape: {out['shape']}")

    print()


def print_model_transform_command(
    model_name,
    onnx_path,
    input_shape,
    preprocess,
    keep_aspect_ratio,
    output_names,
    test_input,
    test_result,
    mlir_path,
):
    parts = [
        ("kv", "--model_name", model_name),
        ("kv", "--model_def", onnx_path),
        ("kv", "--input_shapes", shape_to_tpu_mlir(input_shape)),
        ("kv", "--mean", comma_list(preprocess["mean"])),
        ("kv", "--scale", comma_list(preprocess["scale"])),
    ]

    if keep_aspect_ratio:
        parts.append(("flag", "--keep_aspect_ratio", None))

    parts.append(("kv", "--pixel_format", preprocess["pixel_format"]))

    if output_names:
        parts.append(("kv", "--output_names", ",".join(output_names)))

    if test_input:
        parts.append(("kv", "--test_input", test_input))

    if test_result:
        parts.append(("kv", "--test_result", test_result))

    parts.append(("kv", "--mlir", mlir_path))

    print("# ----- model_transform -----")
    print("model_transform \\")

    for idx, item in enumerate(parts):
        item_type, key, value = item
        is_last = idx == len(parts) - 1

        if item_type == "flag":
            line = f"    {key}"
        else:
            line = f"    {key} {value}"

        if not is_last:
            line += " \\"

        print(line)

    print()


def print_model_deploy_command(
    mlir_path,
    quantize,
    processor,
    test_input,
    test_reference,
    bmodel_path,
    calibration_table=None,
):
    quantize_upper = quantize.upper()

    parts = [
        ("kv", "--mlir", mlir_path),
        ("kv", "--quantize", quantize_upper),
    ]

    if quantize_upper == "INT8":
        if calibration_table is None:
            raise RuntimeError(
                "INT8 requires calibration table.\n"
                "Set DET_CALIBRATION_TABLE or REC_CALIBRATION_TABLE."
            )

        parts.append(("kv", "--calibration_table", calibration_table))

    parts.extend([
        ("kv", "--processor", processor),
        ("kv", "--test_input", test_input),
        ("kv", "--test_reference", test_reference),
        ("kv", "--model", bmodel_path),
    ])

    print("# ----- model_deploy -----")
    print("model_deploy \\")

    for idx, item in enumerate(parts):
        _, key, value = item
        is_last = idx == len(parts) - 1

        line = f"    {key} {value}"

        if not is_last:
            line += " \\"

        print(line)

    print()


# ============================================================
# MODEL GENERATION
# ============================================================

def generate_for_model(model_type):
    if model_type == "det":
        config_path = DET_CONFIG_PATH
        onnx_path = DET_ONNX_PATH
        model_name = DET_MODEL_NAME
        mlir_path = DET_MLIR_PATH
        test_input = DET_TEST_INPUT
        test_result = DET_TEST_RESULT
        manual_shape = DET_INPUT_SHAPE
        keep_aspect_ratio = DET_KEEP_ASPECT_RATIO
        output_names = DET_OUTPUT_NAMES

        deploy_test_input = DET_DEPLOY_TEST_INPUT
        bmodel_path = DET_BMODEL_PATH
        calibration_table = DET_CALIBRATION_TABLE

    elif model_type == "rec":
        config_path = REC_CONFIG_PATH
        onnx_path = REC_ONNX_PATH
        model_name = REC_MODEL_NAME
        mlir_path = REC_MLIR_PATH
        test_input = REC_TEST_INPUT
        test_result = REC_TEST_RESULT
        manual_shape = REC_INPUT_SHAPE
        keep_aspect_ratio = REC_KEEP_ASPECT_RATIO
        output_names = REC_OUTPUT_NAMES

        deploy_test_input = REC_DEPLOY_TEST_INPUT
        bmodel_path = REC_BMODEL_PATH
        calibration_table = REC_CALIBRATION_TABLE

    else:
        raise RuntimeError(f"Unsupported model_type: {model_type}")

    config_path_obj = Path(config_path)
    onnx_path_obj = Path(onnx_path)

    if not config_path_obj.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if not onnx_path_obj.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    config = load_yaml(config_path)
    onnx_info = get_onnx_info(onnx_path)

    if model_type == "det":
        preprocess = get_det_preprocess(config)
    else:
        preprocess = get_rec_preprocess(config)

    input_shape = resolve_input_shape(
        model_type=model_type,
        config=config,
        onnx_info=onnx_info,
        manual_shape=manual_shape,
    )

    print_model_info(
        model_type=model_type,
        config_path=config_path,
        onnx_path=onnx_path,
        onnx_info=onnx_info,
        input_shape=input_shape,
        preprocess=preprocess,
    )

    print_model_transform_command(
        model_name=model_name,
        onnx_path=onnx_path,
        input_shape=input_shape,
        preprocess=preprocess,
        keep_aspect_ratio=keep_aspect_ratio,
        output_names=output_names,
        test_input=test_input,
        test_result=test_result,
        mlir_path=mlir_path,
    )

    if PRINT_MODEL_DEPLOY:
        print_model_deploy_command(
            mlir_path=mlir_path,
            quantize=QUANTIZE,
            processor=PROCESSOR,
            test_input=deploy_test_input,
            test_reference=test_result,
            bmodel_path=bmodel_path,
            calibration_table=calibration_table,
        )


# ============================================================
# MENU
# ============================================================

def model_menu():
    print("#### Model lists ####")
    print("1) rec")
    print("2) det")
    print("3) both")
    print("4) exit")


def ask_target_model():
    while True:
        model_menu()
        choice = input("Enter model: ").strip().lower()

        if choice in {"1", "rec"}:
            return "rec"

        if choice in {"2", "det"}:
            return "det"

        if choice in {"3", "both"}:
            return "both"

        if choice in {"4", "exit", "q", "quit"}:
            return "exit"

        print("Invalid input. Please enter 1, 2, 3, or 4.\n")


# ============================================================
# MAIN
# ============================================================

def run_target(target_model):
    if target_model == "both":
        generate_for_model("rec")
        generate_for_model("det")

    elif target_model in {"rec", "det"}:
        generate_for_model(target_model)

    elif target_model == "exit":
        print("Exit program.")

    else:
        raise RuntimeError("TARGET_MODEL must be 'rec', 'det', 'both', None, or 'exit'.")


def main():
    if TARGET_MODEL is not None:
        run_target(TARGET_MODEL)
        return

    while True:
        target_model = ask_target_model()

        if target_model == "exit":
            print("Exit program.")
            break

        run_target(target_model)
        print()


if __name__ == "__main__":
    main()
