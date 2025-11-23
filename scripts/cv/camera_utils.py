# scripts/cv/camera_utils.py

import cv2


def open_camera(index=0, width=640, height=480):
    """
    打开摄像头
    参数：
        index: 摄像头编号（默认0）
        width, height: 设置分辨率
    返回：
        打开的 cv2.VideoCapture 对象
    """
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        raise RuntimeError("❌ 摄像头无法打开，请检查设备或权限")

    print(f"✅ 摄像头已打开，分辨率：{width}x{height}")
    return cap


def read_frame(cap, flip=True):
    """
    从摄像头读取一帧图像
    参数：
        cap: 已打开的 VideoCapture 对象
        flip: 是否水平翻转（默认 True，使左右一致）
    返回：
        当前帧的图像（BGR 格式）
    """
    ret, frame = cap.read()
    if not ret:
        return None
    if flip:
        frame = cv2.flip(frame, 1)
    return frame


def release_camera(cap):
    """
    释放摄像头资源并关闭窗口
    """
    cap.release()
    cv2.destroyAllWindows()
    print("📷 摄像头已关闭，窗口已销毁")
