import cv2
from models.deepface import *
import os

BORDER_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


def main() -> None:
    os.remove("db/representations_vgg_face.pkl")
    print("Removed VGG-Face Representations pickle file!")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = FaceDetector.detect_faces(img=frame, **FACE_DETECTOR_PARAMS)

        target_face = None
        for face, (x, y, w, h) in faces:
            target_face = face
            cv2.rectangle(frame, (x, y - 50),
                          (x + w, y + h + 10), BORDER_COLOR, 2)

            facial_features = DeepFace.analyze(
                face, **FACIAL_FEATURE_ANALYSIS_PARAMS)
            extracted_facial_features = extract_facial_feature_info(
                facial_features)

            cv2.rectangle(frame, (x - 2, y - 80), (x + (w * 3) // 4, y - 50),
                          BORDER_COLOR, cv2.FILLED)

            matches = DeepFace.find(face, **FACE_RECOGNITION_PARAMS)
            if len(matches):
                name = extract_face_recognition_data(matches.iloc[0])
                cv2.putText(frame, f"Identity: {name}", (x+2, y-70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Identity: Unknown", (x+2, y-70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1, cv2.LINE_AA)

            for i, features in enumerate(extracted_facial_features):
                cv2.putText(frame, features, (x+2, y-55+i*20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1, cv2.LINE_AA)

        cv2.imshow('Face Expression Detection', frame)

        key_pressed = cv2.waitKey(33)
        if key_pressed == ord('q'):
            break
        elif key_pressed == ord('s'):
            name = input("Name of the person: ")
            cv2.imwrite(f"./db/{name}.png", target_face)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
