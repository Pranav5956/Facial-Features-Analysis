import cv2
from models.deepface import *


def main() -> None:
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = FaceDetector.detect_faces(img=frame, **FACE_DETECTOR_PARAMS)

        for face, (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)

            facial_features = DeepFace.analyze(
                face, **FACIAL_FEATURE_ANALYSIS_PARAMS)
            extracted_facial_features = extract_facial_feature_info(
                facial_features)

            for i, features in enumerate(extracted_facial_features):
                cv2.putText(frame, features, (x+20, y-60+i*10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Face Expression Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
