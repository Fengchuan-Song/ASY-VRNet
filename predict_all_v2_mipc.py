import argparse
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from utils.utils import cvtColor, get_classes, preprocess_input, preprocess_input_radar, resize_image
from utils.utils_bbox import non_max_suppression
from utils_seg.utils import resize_image as resize_image_seg

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes", "y")


def list_image_ids(args):
    if args.annotation_path:
        with open(args.annotation_path, "r", encoding="utf-8") as f:
            return [Path(line.strip().split()[0]).stem for line in f if line.strip()]

    image_dir = Path(args.image_dir)
    return sorted(p.stem for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def find_image(image_dir, image_id):
    image_dir = Path(image_dir)
    for suffix in IMAGE_SUFFIXES:
        image_path = image_dir / f"{image_id}{suffix}"
        if image_path.exists():
            return image_path
    return None


def install_optional_import_stubs():
    # EfficientVRNet imports these profiling helpers at module import time,
    # but prediction does not need them.
    try:
        import thop  # noqa: F401
    except ImportError:
        thop_stub = types.ModuleType("thop")
        thop_stub.profile = lambda *args, **kwargs: None
        thop_stub.clever_format = lambda values, *args, **kwargs: values
        sys.modules["thop"] = thop_stub

    try:
        import torchinfo  # noqa: F401
    except ImportError:
        torchinfo_stub = types.ModuleType("torchinfo")
        torchinfo_stub.summary = lambda *args, **kwargs: None
        sys.modules["torchinfo"] = torchinfo_stub


def load_model(args, num_classes, device):
    install_optional_import_stubs()
    from nets.efficient_vrnet import EfficientVRNet

    model = EfficientVRNet(
        num_classes=num_classes,
        num_seg_classes=args.num_classes_seg,
        phi=args.phi,
    )

    state_dict = torch.load(args.model_path, map_location=device)
    if isinstance(state_dict, dict):
        for key in ("state_dict", "model", "ema"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=parse_bool(args.strict_load))
    model.to(device)
    model.eval()
    return model


def decode_outputs_device(outputs, input_shape, device):
    grids = []
    strides = []
    hw = [x.shape[-2:] for x in outputs]
    outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
    outputs[:, :, 4:] = torch.sigmoid(outputs[:, :, 4:])

    for h, w in hw:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        grid = torch.stack((grid_x, grid_y), 2).view(1, -1, 2)
        shape = grid.shape[:2]
        grids.append(grid)
        strides.append(torch.full((shape[0], shape[1], 1), input_shape[0] / h, device=device))

    grids = torch.cat(grids, dim=1).type(outputs.type())
    strides = torch.cat(strides, dim=1).type(outputs.type())
    outputs[..., :2] = (outputs[..., :2] + grids) * strides
    outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
    outputs[..., [0, 2]] = outputs[..., [0, 2]] / input_shape[1]
    outputs[..., [1, 3]] = outputs[..., [1, 3]] / input_shape[0]
    return outputs


def nms_device(prediction, num_classes, input_shape, image_shape, letterbox_image, confidence, nms_iou):
    results = non_max_suppression(
        prediction,
        num_classes,
        input_shape,
        image_shape,
        letterbox_image,
        conf_thres=confidence,
        nms_thres=nms_iou,
    )
    return results[0]


def prepare_image(image, input_shape, letterbox_image):
    image = cvtColor(image)
    image_data = resize_image(image, (input_shape[1], input_shape[0]), letterbox_image)
    _, nw, nh = resize_image_seg(image, (input_shape[1], input_shape[0]))
    image_data = np.expand_dims(
        np.transpose(preprocess_input(np.array(image_data, dtype=np.float32)), (2, 0, 1)),
        0,
    )
    return image, image_data, nw, nh


def prepare_radar(radar_path, radar_channels, normalize_radar):
    radar_data = np.load(radar_path)["arr_0"]
    if radar_data.ndim == 3 and radar_data.shape[0] != radar_channels and radar_data.shape[-1] == radar_channels:
        radar_data = np.transpose(radar_data, (2, 0, 1))
    radar_data = radar_data.astype(np.float32)
    if normalize_radar:
        radar_data = preprocess_input_radar(radar_data)
    return np.expand_dims(radar_data, 0)


def postprocess_seg(output_seg, original_size, input_shape, nw, nh):
    original_w, original_h = original_size
    output_seg = F.softmax(output_seg[0].permute(1, 2, 0), dim=-1).cpu().numpy()
    output_seg = output_seg[
        int((input_shape[0] - nh) // 2): int((input_shape[0] - nh) // 2 + nh),
        int((input_shape[1] - nw) // 2): int((input_shape[1] - nw) // 2 + nw),
    ]
    output_seg = cv2.resize(output_seg, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    return output_seg.argmax(axis=-1).astype(np.uint8)


def save_voc_segmentation_mask(mask, save_path):
    Image.fromarray(mask.astype(np.uint8), mode="L").save(save_path)


def draw_detections(image, detections, class_names, colors, input_shape):
    if detections is None:
        return image

    font_size = int(np.floor(3e-2 * image.size[1] + 0.5))
    try:
        font = ImageFont.truetype(font="model_data/simhei.ttf", size=font_size)
    except OSError:
        font = ImageFont.load_default()

    thickness = int(max((image.size[0] + image.size[1]) // np.mean(input_shape), 1))
    draw = ImageDraw.Draw(image)
    top_label = np.array(detections[:, 6], dtype=np.int32)
    top_conf = detections[:, 4] * detections[:, 5]
    top_boxes = detections[:, :4]

    for i, cls_id in enumerate(top_label):
        predicted_class = class_names[int(cls_id)]
        score = top_conf[i]
        top, left, bottom, right = top_boxes[i]
        top = max(0, int(np.floor(top)))
        left = max(0, int(np.floor(left)))
        bottom = min(image.size[1], int(np.floor(bottom)))
        right = min(image.size[0], int(np.floor(right)))
        label = f"{predicted_class} {score:.2f}"

        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            label_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        except AttributeError:
            label_size = draw.textsize(label, font)
        text_origin = np.array([left, top - label_size[1]]) if top - label_size[1] >= 0 else np.array([left, top + 1])

        color = colors[int(cls_id) % len(colors)]
        for t in range(thickness):
            draw.rectangle([left + t, top + t, right - t, bottom - t], outline=color)
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=color)
        draw.text(tuple(text_origin), label, fill=(0, 0, 0), font=font)
    return image


def save_detection_txt(txt_path, detections):
    with open(txt_path, "w", encoding="utf-8") as f:
        if detections is None:
            return
        top_label = np.array(detections[:, 6], dtype=np.int32)
        top_conf = detections[:, 4] * detections[:, 5]
        top_boxes = detections[:, :4]
        for i, cls_id in enumerate(top_label):
            score = top_conf[i]
            top, left, bottom, right = top_boxes[i]
            f.write(
                "{} {:.6f} {} {} {} {}\n".format(
                    int(cls_id),
                    float(score),
                    int(left),
                    int(top),
                    int(right),
                    int(bottom),
                )
            )


def make_det_colors(num_classes):
    colors = []
    for i in range(max(num_classes, 1)):
        hue = int(180 * i / max(num_classes, 1))
        rgb = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2RGB)[0, 0]
        colors.append(tuple(int(v) for v in rgb))
    return colors


def parse_args():
    parser = argparse.ArgumentParser(description="Predict detection and segmentation with ASY-VRNET/EfficientVRNet.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/EfficientVRNet/weights/Efficient-VRNet_weight.pth",
        help="Path to .pth weights.",
    )
    parser.add_argument("--data_root", type=str, default="/data_ssd/datasets/WaterScenes")
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--radar_dir", type=str, default=None)
    parser.add_argument("--annotation_path", type=str, default='/data_ssd/datasets/WaterScenes/MIPC_shipOnly/2007_test.txt', help="Optional txt; first token is image path/id.")
    parser.add_argument("--save_dir", type=str, default="/data/EfficientVRNet/predict_results")
    parser.add_argument("--classes_path", type=str, default="model_data/waterscenes_benchmark_ship_only.txt")
    parser.add_argument("--cuda", type=str, default="True")
    parser.add_argument("--phi", type=str, default="nano")
    parser.add_argument("--resolution", type=int, default=320)
    parser.add_argument("--rader_channels", type=int, default=4)
    parser.add_argument("--num_classes_seg", type=int, default=9)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--nms_iou", type=float, default=0.35)
    parser.add_argument("--letterbox_image", type=str, default="True")
    parser.add_argument("--normalize_radar", type=str, default="False")
    parser.add_argument("--strict_load", type=str, default="True")
    parser.add_argument("--save_det_image", action="store_true", help="Also save images with detection boxes.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.image_dir = args.image_dir or os.path.join(args.data_root, "images")
    args.radar_dir = args.radar_dir or os.path.join(args.data_root, "radar", "VOCradar320_v2")
    if args.annotation_path is None:
        default_annotation_path = os.path.join(args.data_root, "MIPC_shipOnly", "2007_test.txt")
        args.annotation_path = default_annotation_path if os.path.exists(default_annotation_path) else None

    input_shape = [args.resolution, args.resolution]
    cuda = parse_bool(args.cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    class_names, num_classes = get_classes(args.classes_path)
    model = load_model(args, num_classes, device)
    det_colors = make_det_colors(num_classes)

    save_dir = Path(args.save_dir)
    det_txt_dir = save_dir / "DetectionResults"
    seg_mask_dir = save_dir / "SegmentationClass"
    det_img_dir = save_dir / "detection-images"
    det_txt_dir.mkdir(parents=True, exist_ok=True)
    seg_mask_dir.mkdir(parents=True, exist_ok=True)
    if args.save_det_image:
        det_img_dir.mkdir(parents=True, exist_ok=True)

    image_ids = list_image_ids(args)
    iterator = tqdm(image_ids, desc="Predicting") if tqdm is not None else image_ids
    skipped = 0

    for image_id in iterator:
        image_path = find_image(args.image_dir, image_id)
        radar_path = Path(args.radar_dir) / f"{image_id}.npz"
        if image_path is None or not radar_path.exists():
            skipped += 1
            print(f"Skip {image_id}: missing image or radar npz.")
            continue

        image = Image.open(image_path)
        original_image, image_data, nw, nh = prepare_image(image, input_shape, parse_bool(args.letterbox_image))
        radar_data = prepare_radar(radar_path, args.rader_channels, parse_bool(args.normalize_radar))
        image_shape = np.array(np.shape(original_image)[0:2])

        with torch.no_grad():
            images = torch.from_numpy(image_data).to(device)
            radars = torch.from_numpy(radar_data).float().to(device)
            det_outputs, seg_output = model(images, radars)
            det_outputs = decode_outputs_device(det_outputs, input_shape, device)
            detections = nms_device(
                det_outputs,
                num_classes,
                input_shape,
                image_shape,
                parse_bool(args.letterbox_image),
                args.confidence,
                args.nms_iou,
            )
            seg_mask = postprocess_seg(seg_output, original_image.size, input_shape, nw, nh)

        save_detection_txt(det_txt_dir / f"{image_id}.txt", detections)
        save_voc_segmentation_mask(seg_mask, seg_mask_dir / f"{image_id}.png")

        if args.save_det_image:
            det_image = draw_detections(original_image.copy(), detections, class_names, det_colors, input_shape)
            det_image.save(det_img_dir / f"{image_id}.jpg", quality=95, subsampling=0)

    print(f"Saved detection txt to: {det_txt_dir}")
    print(f"Saved VOC-style segmentation class masks to: {seg_mask_dir}")
    if args.save_det_image:
        print(f"Saved detection images to: {det_img_dir}")
    if skipped:
        print(f"Skipped {skipped} samples because image or radar file was missing.")


if __name__ == "__main__":
    main()
