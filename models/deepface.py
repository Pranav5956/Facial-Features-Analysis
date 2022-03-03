from deepface import DeepFace
from deepface.detectors import FaceDetector

# Face Detection
FACE_DETECTOR_BACKEND = "ssd"
FACE_DETECTOR_MODEL = FaceDetector.build_model(FACE_DETECTOR_BACKEND)
FACE_DETECTOR_PARAMS = {
    'face_detector': FACE_DETECTOR_MODEL,
    'detector_baclend': FACE_DETECTOR_BACKEND,
    'align': True
}

# Facial Feature Analysiss
FACIAL_FEATURES = ['emotion']
FACIAL_FEATURE_ANALYSIS_MODELS = {
    feature: DeepFace.build_model(feature.capitalize())
    for feature in FACIAL_FEATURES
}
FACIAL_FEATURE_ANALYSIS_PARAMS = {
    'models': FACIAL_FEATURE_ANALYSIS_MODELS,
    'actions': FACIAL_FEATURES,
    'detector_backend': FACE_DETECTOR_BACKEND,
    'enforce_detection': False,
    'prog_bar': False
}


def extract_facial_feature_info(features):
    extracted_features = []

    for feature in FACIAL_FEATURES:
        if feature in ["age", "gender"]:
            extracted_features.append(
                f"{feature.capitalize()}: {features[feature]}")
        else:
            extracted_features.append(
                f"{feature.capitalize()}: {features['dominant_' + feature]}")

    return extracted_features
