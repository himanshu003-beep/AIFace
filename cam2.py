import cv2
import time
import os
import torch
import csv
from datetime import datetime
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# 1. Directory and file setup
OUTPUT_DIR = "cropped_faces"
LOG_FILE = "attendance_log.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize CSV log file if not present
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Person_ID", "Action", "Image_File"])

# 2. Hardware acceleration and model initialization
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[*] Initializing models on device: {device}")

mtcnn = MTCNN(keep_all=True, min_face_size=40, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# Registry structure:
# { person_id: { "embedding": tensor, "status": "LOGGED_IN"/"LOGGED_OUT", "last_action_time": float, "last_image": path } }
registered_users = {}
person_counter = 1
COOLDOWN_SECONDS = 10  # Cooldown between Login and Logout toggles


def compute_embedding(pil_face_crop):
    """Generates 512-dimensional embedding vector for facial recognition."""
    try:
        face_tensor = pil_face_crop.resize((160, 160))
        tensor_data = torch.tensor(np.array(face_tensor)).permute(2, 0, 1).float()
        tensor_data = (tensor_data - 127.5) / 128.0
        tensor_data = tensor_data.unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = resnet(tensor_data)
        return embedding
    except Exception as e:
        print(f"[!] Embedding Generation Error: {e}")
        return None


def log_event(person_id, action, image_filename):
    """Records event in CSV file."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([now_str, person_id, action, image_filename])


def process_face_action(crop_img, embedding, similarity_threshold=0.65):
    """
    Checks if person exists in registry.
    Toggles between LOGIN and LOGOUT after cooldown.
    Auto-purges old crop and replaces with latest frame.
    """
    global registered_users, person_counter
    current_time = time.time()
    matched_id = None
    highest_similarity = -1.0

    for pid, data in registered_users.items():
        sim = torch.nn.functional.cosine_similarity(embedding, data["embedding"]).item()
        if sim > similarity_threshold and sim > highest_similarity:
            highest_similarity = sim
            matched_id = pid

    timestamp_ms = int(current_time * 1000)

    # Case A: Existing user recognized
    if matched_id is not None:
        user_info = registered_users[matched_id]
        time_elapsed = current_time - user_info["last_action_time"]

        # If still within cooldown, just keep tracking without toggling state
        if time_elapsed < COOLDOWN_SECONDS:
            return matched_id, user_info["status"], f"Wait {int(COOLDOWN_SECONDS - time_elapsed)}s"

        # Cooldown passed -> Toggle state (LOGIN -> LOGOUT or LOGOUT -> LOGIN)
        new_status = "LOGGED_OUT" if user_info["status"] == "LOGGED_IN" else "LOGGED_IN"
        new_filename = f"{matched_id}_{new_status.lower()}_{timestamp_ms}.jpg"
        new_filepath = os.path.join(OUTPUT_DIR, new_filename)

        # Auto-purge previous image file
        old_path = user_info["last_image"]
        if old_path and os.path.exists(old_path):
            try:
                os.remove(old_path)
                print(f"[-] Replaced old record: '{os.path.basename(old_path)}'")
            except Exception as e:
                print(f"[!] Error deleting old face: {e}")

        # Save new updated crop
        crop_img.save(new_filepath)

        # Update registry
        user_info["status"] = new_status
        user_info["last_action_time"] = current_time
        user_info["last_image"] = new_filepath
        user_info["embedding"] = embedding

        log_event(matched_id, new_status, new_filename)
        print(f"[*] [{matched_id}] {new_status} triggered! (Saved: {new_filename})")
        return matched_id, new_status, "Success"

    # Case B: Completely new face -> Initial LOGIN
    else:
        new_id = f"Person_{person_counter}"
        person_counter += 1
        new_status = "LOGGED_IN"
        new_filename = f"{new_id}_login_{timestamp_ms}.jpg"
        new_filepath = os.path.join(OUTPUT_DIR, new_filename)

        crop_img.save(new_filepath)

        registered_users[new_id] = {
            "embedding": embedding,
            "status": new_status,
            "last_action_time": current_time,
            "last_image": new_filepath,
        }

        log_event(new_id, new_status, new_filename)
        print(f"[+] [{new_id}] First detection -> LOGGED_IN! (Saved: {new_filename})")
        return new_id, new_status, "Success"


# 3. Video Capture Setup (Single Front Laptop Camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[!] Error: Front camera could not be opened.")
    exit()

print("=" * 65)
print("[*] Front Camera Login/Logout Attendance Pipeline Active.")
print("[*] Face appears first time -> Auto-LOGIN.")
print(f"[*] Face appears again after {COOLDOWN_SECONDS}s -> Auto-LOGOUT.")
print(f"[*] CSV Logs saved to: '{LOG_FILE}'")
print("[*] Press 'q' to terminate.")
print("=" * 65)

frame_count = 0
active_detections = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("[!] Frame acquisition failed.")
        break

    frame_count += 1

    # Run AI inference every 5th frame for smooth FPS
    if frame_count % 5 == 0:
        active_detections.clear()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        img_w, img_h = pil_img.size

        # type: ignore silences VS Code Pylance lint warnings
        boxes, probs = mtcnn.detect(pil_img, landmarks=False)  # type: ignore

        if boxes is not None and len(boxes) > 0:
            for box, prob in zip(boxes, probs):
                if prob is None or prob < 0.85:
                    continue

                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                face_crop = pil_img.crop((x1, y1, x2, y2))
                embedding = compute_embedding(face_crop)

                if embedding is not None:
                    user_id, status, msg = process_face_action(face_crop, embedding)
                    active_detections.append({
                        "box": (x1, y1, x2, y2),
                        "id": user_id,
                        "status": status,
                        "msg": msg
                    })

    # Render bounding boxes and UI status
    for item in active_detections:
        x1, y1, x2, y2 = item["box"]
        status = item["status"]
        user_id = item["id"]
        msg = item["msg"]

        # Color coding: Green for LOGIN, Red for LOGOUT
        box_color = (0, 255, 0) if status == "LOGGED_IN" else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(
            frame,
            f"{user_id}: {status} [{msg}]",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            box_color,
            2,
        )

    # Top overlay header
    cv2.putText(
        frame,
        "Front Cam Login/Logout System | Press 'q' to Exit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2,
    )

    cv2.imshow("AI Face Recognition - Front Camera Auth", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("[*] Stream terminated cleanly.")