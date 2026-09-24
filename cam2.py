import cv2
import os
import time
import threading
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

LOGIN_DIR = "login_faces"
LOGOUT_DIR = "logout_faces"

for folder in [LOGIN_DIR, LOGOUT_DIR]:
    os.makedirs(folder, exist_ok=True)
    keep_path = os.path.join(folder, ".gitkeep")
    if not os.path.exists(keep_path):
        try:
            with open(keep_path, "w") as f:
                f.write("")
        except Exception:
            pass

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[*] Initializing deep learning models on device: {device}")

mtcnn = MTCNN(
    keep_all=True,
    min_face_size=24,
    thresholds=[0.55, 0.65, 0.65],
    post_process=True,
    device=device
)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

registered_users = {}
person_counter = 1


class FastRTSPStream:
    def __init__(self, src):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|"
            "analyzeduration;1000000|"
            "probesize;1000000|"
            "fflags;nobuffer|flags;low_delay|reorder_queue_size;0|"
            "max_delay;500000"
        )
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()

        if self.cap.isOpened():
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()


def compute_sharpness(bgr_crop):
    if bgr_crop is None or bgr_crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def enhance_crop_quality(raw_crop, target_min_dim=360):
    if raw_crop is None or raw_crop.size == 0:
        return raw_crop

    h, w = raw_crop.shape[:2]
    if h < target_min_dim or w < target_min_dim:
        scale = max(target_min_dim / float(h), target_min_dim / float(w))
        new_w = int(w * scale)
        new_h = int(h * scale)
        enhanced = cv2.resize(raw_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        enhanced = raw_crop.copy()

    gaussian = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(enhanced, 1.25, gaussian, -0.25, 0)
    return sharpened


def get_pristine_face_crop(raw_frame, box, margin_ratio=0.35):
    h_frame, w_frame, _ = raw_frame.shape
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    pad_w = int(w * margin_ratio)
    pad_h = int(h * margin_ratio)

    cx1 = max(0, x1 - pad_w)
    cy1 = max(0, y1 - pad_h)
    cx2 = min(w_frame, x2 + pad_w)
    cy2 = min(h_frame, y2 + pad_h)

    crop = raw_frame[cy1:cy2, cx1:cx2]
    return enhance_crop_quality(crop)


def compute_embedding(pil_face_crop):
    try:
        face_tensor = pil_face_crop.resize((160, 160), Image.Resampling.LANCZOS)
        tensor_data = torch.tensor(np.array(face_tensor)).permute(2, 0, 1).float()
        tensor_data = (tensor_data - 127.5) / 128.0
        tensor_data = tensor_data.unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = resnet(tensor_data)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding
    except Exception as e:
        print(f"[!] Embedding calculation failed: {e}")
        return None


def match_identity(embedding, similarity_threshold=0.48):
    matched_id = None
    highest_sim = -1.0

    for pid, data in registered_users.items():
        for stored_emb in data["embeddings"]:
            sim = torch.nn.functional.cosine_similarity(embedding, stored_emb).item()
            if sim > similarity_threshold and sim > highest_sim:
                highest_sim = sim
                matched_id = pid

    return matched_id, highest_sim


# =====================================================================
# MODIFICATION: Independent Dual-Camera Auto-Enrollment & Single-Save Engine
# Changes made:
# 1. Removed requirement for prior login: anyone appearing at the logout camera
#    is immediately enrolled and their image is saved to logout_faces/.
# 2. Existing users on the logout camera have their image saved independently
#    without requiring an active LOGGED_IN state flag.
# =====================================================================
def register_or_update(raw_crop, embedding, camera_type):
    global registered_users, person_counter

    matched_id, score = match_identity(embedding)
    sharpness_score = compute_sharpness(raw_crop)
    MIN_SHARPNESS = 70.0 if camera_type == "LOGIN" else 35.0

    # 1. New Person Seen for the First Time (on Either Camera)
    if matched_id is None:
        if sharpness_score < MIN_SHARPNESS:
            return "Scanning...", "BLURRY"

        matched_id = f"Person_{person_counter}"
        person_counter += 1

        is_login = (camera_type == "LOGIN")
        registered_users[matched_id] = {
            "embeddings": [embedding],
            "current_status": camera_type,
            "login_saved": is_login,
            "logout_saved": not is_login,
            "last_seen": time.time()
        }

        target_dir = LOGIN_DIR if is_login else LOGOUT_DIR
        save_path = os.path.join(target_dir, f"{matched_id}.jpg")
        cv2.imwrite(save_path, raw_crop, [cv2.IMWRITE_JPEG_QUALITY, 100, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
        print(f"[+] [{camera_type} INDEPENDENT ENROLL] {matched_id} -> {save_path} (Sharpness: {sharpness_score:.1f})")
        return matched_id, camera_type

    # 2. Existing Recognized Identity
    user_info = registered_users[matched_id]
    user_info["last_seen"] = time.time()

    if len(user_info["embeddings"]) < 5 and sharpness_score >= MIN_SHARPNESS:
        user_info["embeddings"].append(embedding)

    # ------------------ LOGIN CAMERA ------------------
    if camera_type == "LOGIN":
        if not user_info["login_saved"]:
            if sharpness_score >= MIN_SHARPNESS:
                save_path = os.path.join(LOGIN_DIR, f"{matched_id}.jpg")
                cv2.imwrite(save_path, raw_crop, [cv2.IMWRITE_JPEG_QUALITY, 100, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
                user_info["login_saved"] = True
                user_info["logout_saved"] = False
                user_info["current_status"] = "LOGIN"
                print(f"[*] [LOGIN SAVED] {matched_id} -> {save_path} (Sharpness: {sharpness_score:.1f})")
        return matched_id, "LOGIN"

    # ------------------ LOGOUT CAMERA ------------------
    elif camera_type == "LOGOUT":
        # Save logout image without requiring prior login confirmation
        if not user_info["logout_saved"]:
            if sharpness_score >= MIN_SHARPNESS:
                save_path = os.path.join(LOGOUT_DIR, f"{matched_id}.jpg")
                cv2.imwrite(save_path, raw_crop, [cv2.IMWRITE_JPEG_QUALITY, 100, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
                user_info["logout_saved"] = True
                user_info["login_saved"] = False
                user_info["current_status"] = "LOGOUT"
                print(f"[-] [LOGOUT SAVED] {matched_id} -> {save_path} (Sharpness: {sharpness_score:.1f})")
        return matched_id, "LOGOUT"

    return matched_id, user_info["current_status"]


def process_feed(frame, camera_type, detections_cache):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    img_w, img_h = pil_img.size

    boxes, probs = mtcnn.detect(pil_img, landmarks=False)  # type: ignore

    detections_cache.clear()
    if boxes is not None and len(boxes) > 0:
        for box, prob in zip(boxes, probs):
            min_prob = 0.75 if camera_type == "LOGIN" else 0.65
            if prob is None or prob < min_prob:
                continue

            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            norm_crop = pil_img.crop((x1, y1, x2, y2))
            emb = compute_embedding(norm_crop)

            if emb is not None:
                raw_crop = get_pristine_face_crop(frame, (x1, y1, x2, y2))
                user_id, status = register_or_update(raw_crop, emb, camera_type)
                detections_cache.append({"box": (x1, y1, x2, y2), "id": user_id, "status": status})


RTSP_URL = "rtsp://nidhin:Nidhin123@192.168.2.178:554"

print("[*] Initializing Laptop Camera (Entry / Login)...")
cap_login = cv2.VideoCapture(0)

print(f"[*] Initializing RTSP Camera (Exit / Logout): {RTSP_URL}...")
cap_logout = FastRTSPStream(RTSP_URL)

if not cap_login.isOpened():
    print("[!] Warning: Laptop front camera unavailable.")

if not cap_logout.isOpened():
    print("[!] Warning: RTSP camera unavailable.")

frame_count = 0
login_detections = []
logout_detections = []

print("=" * 65)
print("[*] Independent Dual-Camera Pipeline Online:")
print("    - Entry Cam : Saves to 'login_faces/'")
print("    - Exit Cam  : Saves to 'logout_faces/' (Fully independent)")
print("[*] Press 'q' to stop.")
print("=" * 65)

while True:
    ret_in, frame_in = cap_login.read() if cap_login.isOpened() else (False, None)
    ret_out, frame_out = cap_logout.read() if cap_logout.isOpened() else (False, None)

    if not ret_in and not ret_out:
        print("[!] Both camera streams are offline.")
        break

    frame_count += 1
    run_inference = (frame_count % 3 == 0)

    # 1. Login Camera (Laptop Webcam)
    if ret_in and frame_in is not None:
        if run_inference:
            process_feed(frame_in, "LOGIN", login_detections)

        for det in login_detections:
            x1, y1, x2, y2 = det["box"]
            color = (0, 255, 0) if "LOGIN" in det["status"] else (0, 255, 255)
            cv2.rectangle(frame_in, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame_in,
                f"{det['id']} [{det['status']}]",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        cv2.putText(frame_in, "ENTRY FEED (Laptop)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Entry Camera (Login)", frame_in)

    # 2. Logout Camera (RTSP CCTV)
    if ret_out and frame_out is not None:
        if run_inference:
            process_feed(frame_out, "LOGOUT", logout_detections)

        for det in logout_detections:
            x1, y1, x2, y2 = det["box"]
            color = (0, 0, 255) if "LOGOUT" in det["status"] else (0, 165, 255)
            cv2.rectangle(frame_out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame_out,
                f"{det['id']} [{det['status']}]",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        cv2.putText(frame_out, "EXIT FEED (RTSP)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Exit Camera (Logout)", frame_out)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

if cap_login.isOpened():
    cap_login.release()
if cap_logout.isOpened():
    cap_logout.release()
cv2.destroyAllWindows()
print("[*] Shutdown complete.")