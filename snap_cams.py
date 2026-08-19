"""snap_cams.py — grab one frame from each RealSense and save full + policy-crop.

Run on Thor with NO other process using the cameras (close the eval client /
policy pipeline first — RealSense allows only one consumer of a stream). Needs
no server, tunnel, or arm. Shows EXACTLY what the policy sees, and prints each
stream's mean brightness (near 0 = truly black; >~20 = a real image).
"""
import numpy as np
import cv2
import pyrealsense2 as rs
from PIL import Image

CAM_BASE_SERIAL = "215222078407"   # base / scene ("image")
CAM_WRIST_SERIAL = "845112070404"  # wrist ("wrist_image")


def grab(sn):
    p = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(sn)
    cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.yuyv, 30)
    p.start(cfg)
    frames = None
    for _ in range(10):  # let auto-exposure settle
        frames = p.wait_for_frames(timeout_ms=2000)
    f = frames.get_color_frame()
    raw = np.asanyarray(f.get_data())
    bgr = cv2.cvtColor(raw.view(np.uint8).reshape(1080, 1920, 2), cv2.COLOR_YUV2BGR_YUYV)
    p.stop()
    return bgr


def crop_resize(bgr, mode, crop=1080, target=224):
    rgb = bgr[:, :, ::-1]
    img = Image.fromarray(rgb)
    w, h = img.size
    left = 540 if mode == "left_540" else (w - 80) - crop
    return np.asarray(img.crop((left, 0, left + crop, crop)).resize((target, target), Image.LANCZOS), np.uint8)


base = grab(CAM_BASE_SERIAL)
wrist = grab(CAM_WRIST_SERIAL)
Image.fromarray(base[:, :, ::-1]).save("base_full.jpg")
Image.fromarray(wrist[:, :, ::-1]).save("wrist_full.jpg")
Image.fromarray(crop_resize(base, "left_540")).save("base_crop.jpg")
Image.fromarray(crop_resize(wrist, "right")).save("wrist_crop.jpg")
print(f"mean brightness  base={base.mean():.1f}  wrist={wrist.mean():.1f}   (near 0 = black, >~20 = real image)")
print("saved: base_full.jpg  wrist_full.jpg  base_crop.jpg  wrist_crop.jpg")
