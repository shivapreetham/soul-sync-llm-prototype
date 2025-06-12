from fer import FER
import cv2

def detect_emotion_fer(frame):
    """
    Uses the fer library's detector; runs a small CNN on the face region.
    """
    detector = FER(mtcnn=True)  # may download weights first time
    # frame must be RGB for fer
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_emotions(rgb)
    # results is a list of dicts: [{'box':(...), 'emotions':{'angry':0.0, 'happy':0.95,...}}]
    if not results:
        return None
    # Pick first face
    emotions = results[0]["emotions"]
    # Return emotion with highest score
    return max(emotions, key=emotions.get)

# Example usage:
cap = cv2.VideoCapture(0)
detector = FER(mtcnn=True)
while True:
    ret, frame = cap.read()
    if not ret: break
    emo = detect_emotion_fer(frame)
    cv2.putText(frame, str(emo), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("FER Emotion", frame)
    if cv2.waitKey(1)&0xFF==ord('q'): break
cap.release()
cv2.destroyAllWindows()
