import cv2
import time
import os
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# 1. Output directory setup
OUTPUT_DIR = "cropped_faces"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Hardware acceleration and model initialization
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[*] Initializing models on device: {device}")

# Face detector and embedding generator models
mtcnn = MTCNN(keep_all=True, min_face_size=40, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# In-memory tracking cache: stores { "file_path": embedding_tensor }
stored_faces = {}


def compute_embedding(pil_face_crop):
    """Generates a 512-dimensional embedding vector for facial identity matching."""
    try:
        # Resize and normalize tensor for ResNet
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


def match_and_replace_face(new_crop, new_embedding, similarity_threshold=0.65):
    """
    Checks if face exists. If matched, deletes previous file and updates with new crop.
    If new member, saves as a fresh identity.
    """
    global stored_faces
    matched_key = None
    highest_similarity = -1.0

    # Calculate cosine similarity with all previously stored faces
    for old_path, old_embedding in list(stored_faces.items()):
        similarity = torch.nn.functional.cosine_similarity(new_embedding, old_embedding).item()
        if similarity > similarity_threshold and similarity > highest_similarity:
            highest_similarity = similarity
            matched_key = old_path

    timestamp = int(time.time() * 1000)
    new_filename = f"face_{timestamp}.jpg"
    new_filepath = os.path.join(OUTPUT_DIR, new_filename)

    # Scenario A: Existing person returned -> Purge old image, replace with new
    if matched_key is not None:
        try:
            if os.path.exists(matched_key):
                os.remove(matched_key)
                print(f"[-] Replaced identity: Removed '{os.path.basename(matched_key)}'")
        except Exception as e:
            print(f"[!] Deletion error: {e}")

        # Update cache
        del stored_faces[matched_key]
        new_crop.save(new_filepath)
        stored_faces[new_filepath] = new_embedding
        print(f"[+] Updated face saved: {new_filename} (Similarity: {highest_similarity:.2f})")

    # Scenario B: Brand new person detected -> Save new identity
    else:
        new_crop.save(new_filepath)
        stored_faces[new_filepath] = new_embedding
        print(f"[+] New member added: {new_filename}")


# 3. Video source configuration
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[!] Error: Could not access video capture device.")
    exit()

print("=" * 65)
print("[*] Face Tracking & Dynamic Replacement Engine Started.")
print("[*] Existing members: Old face auto-purged, updated with latest frame.")
print("[*] New members: Registered and tracked continuously.")
print("[*] Press 'q' to terminate.")
print("=" * 65)

frame_count = 0
faces_data = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("[!] Frame read failed.")
        break

    frame_count += 1

    # Run detection periodically every 5th frame to maintain smooth FPS
    if frame_count % 5 == 0:
        faces_data = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        img_w, img_h = pil_img.size

        # Added '# type: ignore' to permanently silence VS Code Pylance linting warnings
        boxes, probs = mtcnn.detect(pil_img, landmarks=False)  # type: ignore

        if boxes is not None and len(boxes) > 0:
            for box, prob in zip(boxes, probs):
                if prob is None or prob < 0.90:
                    continue

                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                # Crop face area
                cropped_face = pil_img.crop((x1, y1, x2, y2))
                embedding = compute_embedding(cropped_face)

                if embedding is not None:
                    match_and_replace_face(cropped_face, embedding)

                faces_data.append({
                    "box": (x1, y1, x2, y2),
                    "confidence": float(prob)
                })

    # Render bounding boxes on display feed
    for face in faces_data:
        x1, y1, x2, y2 = face["box"]
        prob = face["confidence"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Active: {prob:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.putText(
        frame,
        "Auto-Replace Mode Active | Press 'q' to Exit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    cv2.imshow("Face Recognition - Identity Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("[*] Stream terminated cleanly.")