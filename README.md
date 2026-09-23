# AI Face Recognition & Automated Attendance System

A real-time facial recognition and identity tracking pipeline built with PyTorch, MTCNN, and InceptionResnetV1 (FaceNet). The system dynamically detects faces, replaces outdated captures with clear frames, and logs user logins and logouts using front camera tracking.

---

## Features

- **Real-Time Detection & Embeddings**: Uses MTCNN for face localization and pretrained InceptionResnetV1 (`vggface2`) to generate 512-dimensional feature vectors.
- **Dynamic Identity Purging**: Uses cosine similarity mapping (> 0.65 threshold). When an existing person is re-identified, their older image is purged from the disk and updated with the latest frame.
- **Automated Login & Logout (`cam2.py`)**:
  - Automatically marks `LOGIN` (Green bounding box) on first detection.
  - Implements a configurable cooldown period (10 seconds) to avoid repeated state flips.
  - Automatically logs `LOGOUT` (Red bounding box) upon subsequent entry after the cooldown.
- **Local Attendance Logging**: Records timestamps, user IDs, actions, and saved image references into `attendance_log.csv`.
- **CPU/GPU Optimized**: Processes detection every 5th frame to avoid FPS lag and reduce compute load.

---

## Tech Stack

- **Python 3.10+**
- **PyTorch** (`torch`, `torchvision`)
- **facenet-pytorch** (MTCNN & InceptionResnetV1)
- **OpenCV** (`opencv-python`)
- **NumPy**
- **Pillow** (`PIL`)

---

## Project Structure

```text
├── cropped_faces/       # Directory where active user face crops are saved/updated
├── main.py              # Real-time face detection & dynamic crop replacement engine
├── cam2.py              # Front camera Login/Logout pipeline with attendance logger
├── requirements.txt     # Pinned Python package dependencies
└── README.md            # Project overview and setup instructions