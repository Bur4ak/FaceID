import cv2
from deepface import DeepFace
import numpy as np
import tempfile
import os

model_name = "ArcFace"
threshold = 3.7  # Burak'ın deneyine göre iyi eşik

# ✅ Kayıtlı yüzler (isim : dosya yolu)
face_db = {
    "Elon": "elon.jpg",
    "Mark": "Mark.jpg",
}

# Her yüzün embedding’ini çıkar
print("🔍 Yüz veritabanı hazırlanıyor...")
embeddings = {}
for name, path in face_db.items():
    emb = DeepFace.represent(path, model_name=model_name,
                             detector_backend="opencv", enforce_detection=False)[0]["embedding"]
    embeddings[name] = np.array(emb)
print("✅ Veritabanı hazır!")

# Kamera başlat
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                temp_path = tmp.name
                cv2.imwrite(temp_path, face_img)

            try:
                emb = DeepFace.represent(temp_path, model_name=model_name,
                                         detector_backend="opencv", enforce_detection=False)[0]["embedding"]
                emb = np.array(emb)

                # Her kayıtlı yüz ile karşılaştır
                distances = {}
                for name, ref_emb in embeddings.items():
                    dist = np.linalg.norm(ref_emb - emb)
                    distances[name] = dist

                # En küçük mesafeyi bul
                best_match = min(distances, key=distances.get)
                best_distance = distances[best_match]

                if best_distance < threshold:
                    label = f"{best_match} ({best_distance:.2f})"
                    color = (0, 255, 0)
                else:
                    label = f"Unknown ({best_distance:.2f})"
                    color = (0, 0, 255)

            except Exception as e:
                label = "Error"
                color = (0, 0, 255)
                print("Hata:", e)

            os.remove(temp_path)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Multi Face ID (q = quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
