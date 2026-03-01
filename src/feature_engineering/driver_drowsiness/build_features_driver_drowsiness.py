"""
Driver Drowsiness Feature Extraction Module

This script extracts spatial and temporal features from preprocessed video sequences for:
1. YawDD dataset (yawning detection)
2. NTHU-DDD dataset (drowsiness detection)

Features extracted include:
- Eye Aspect Ratio (EAR) for blink detection
- Mouth Aspect Ratio (MAR) for yawn detection
- Pupil circularity
- Head pose estimation (simplified)
- Temporal statistics (mean, std, max) over sequences

Output features are saved as numpy arrays for model training.
"""

import os
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class DrowsinessFeatureBuilder:
    """
    Feature extraction pipeline for driver drowsiness detection.
    Integrates with existing preprocessing from driver_drowsiness_detection.ipynb.
    """

    def __init__(self):
        """Initialize MediaPipe models and feature configurations"""
        # MediaPipe Face Mesh for facial landmarks
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Landmark indices for key features (MediaPipe 478-point model)
        self.LANDMARK_INDICES = {
            'left_eye': [33, 160, 158, 133, 153, 144],
            'right_eye': [362, 385, 387, 263, 373, 380],
            'mouth': [61, 291, 39, 181, 0, 17, 269, 405],
            'face_oval': [10, 338, 297, 332, 284, 251, 389, 356]
        }

        # Feature configuration
        self.FEATURE_NAMES = [
            'left_ear', 'right_ear', 'avg_ear',
            'mouth_ratio', 'pupil_circularity',
            'head_tilt', 'head_rotation'
        ]

    def extract_frame_features(self, frame: np.ndarray) -> Dict:
        """
        Extract drowsiness-related features from a single frame.

        Args:
            frame: Input image in BGR format (0-255)

        Returns:
            Dictionary containing:
            - 'landmarks': Raw facial landmarks if needed
            - 'features': Extracted feature vector
            - 'success': Boolean indicating detection success
        """
        results = {
            'landmarks': None,
            'features': None,
            'success': False
        }

        # Convert to RGB and process
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_results = self.face_mesh.process(rgb_frame)

        if not mesh_results.multi_face_landmarks:
            return results

        landmarks = mesh_results.multi_face_landmarks[0].landmark
        results['landmarks'] = [(lm.x, lm.y, lm.z) for lm in landmarks]

        # Calculate all features
        features = []

        # 1. Eye features (EAR and pupil circularity)
        left_eye = [landmarks[i] for i in self.LANDMARK_INDICES['left_eye']]
        right_eye = [landmarks[i] for i in self.LANDMARK_INDICES['right_eye']]

        left_ear = self._calculate_ear(left_eye)
        right_ear = self._calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        pupil_circularity = self._calculate_pupil_circularity(left_eye + right_eye)

        features.extend([left_ear, right_ear, avg_ear])

        # 2. Mouth features
        mouth = [landmarks[i] for i in self.LANDMARK_INDICES['mouth']]
        mouth_ratio = self._calculate_mouth_ratio(mouth)
        features.append(mouth_ratio)

        # 3. Simplified head pose
        face_oval = [landmarks[i] for i in self.LANDMARK_INDICES['face_oval']]
        head_tilt, head_rotation = self._estimate_head_pose(face_oval)
        features.extend([pupil_circularity, head_tilt, head_rotation])

        results['features'] = np.array(features)
        results['success'] = True

        return results

    def _calculate_ear(self, eye_landmarks: List) -> float:
        """Calculate Eye Aspect Ratio (EAR) for blink detection"""
        # Vertical distances
        v1 = np.linalg.norm([eye_landmarks[1].x - eye_landmarks[5].x,
                           eye_landmarks[1].y - eye_landmarks[5].y])
        v2 = np.linalg.norm([eye_landmarks[2].x - eye_landmarks[4].x,
                           eye_landmarks[2].y - eye_landmarks[4].y])

        # Horizontal distance
        h = np.linalg.norm([eye_landmarks[0].x - eye_landmarks[3].x,
                          eye_landmarks[0].y - eye_landmarks[3].y])

        return (v1 + v2) / (2.0 * h + 1e-6)  # Add epsilon to avoid division by zero

    def _calculate_mouth_ratio(self, mouth_landmarks: List) -> float:
        """Calculate Mouth Aspect Ratio (MAR) for yawn detection"""
        # Vertical distance (upper to lower lip)
        vertical = np.linalg.norm([mouth_landmarks[2].x - mouth_landmarks[6].x,
                                 mouth_landmarks[2].y - mouth_landmarks[6].y])

        # Horizontal distance (mouth corners)
        horizontal = np.linalg.norm([mouth_landmarks[0].x - mouth_landmarks[4].x,
                                   mouth_landmarks[0].y - mouth_landmarks[4].y])

        return vertical / (horizontal + 1e-6)

    def _calculate_pupil_circularity(self, eye_landmarks: List) -> float:
        """Estimate pupil circularity (1 = perfect circle)"""
        # Use convex hull area vs actual area
        points = np.array([(lm.x, lm.y) for lm in eye_landmarks])
        area = cv2.contourArea(points)
        hull = cv2.convexHull(points)
        hull_area = cv2.contourArea(hull)
        return area / (hull_area + 1e-6)

    def _estimate_head_pose(self, face_oval: List) -> Tuple[float, float]:
        """Simplified head pose estimation (tilt and rotation)"""
        # Get extreme points
        left = min([lm.x for lm in face_oval])
        right = max([lm.x for lm in face_oval])
        top = min([lm.y for lm in face_oval])
        bottom = max([lm.y for lm in face_oval])

        # Calculate tilt (vertical asymmetry)
        left_height = bottom - top
        right_height = bottom - top  # Simplified for basic estimation
        tilt = (right_height - left_height) / (left_height + 1e-6)

        # Calculate rotation (horizontal asymmetry)
        width = right - left
        rotation = (face_oval[0].y - face_oval[-1].y) / width

        return tilt, rotation

    def extract_sequence_features(self, sequence: np.ndarray) -> np.ndarray:
        """
        Extract temporal features from a sequence of frames.

        Args:
            sequence: Numpy array of shape (seq_len, height, width, channels)

        Returns:
            Numpy array of shape (n_features,) containing:
            - Mean, std, max of each feature across the sequence
            - Temporal derivatives of key features
        """
        frame_features = []

        for frame in sequence:
            # Convert from normalized [0,1] to [0,255]
            frame_uint8 = (frame * 255).astype('uint8')
            results = self.extract_frame_features(frame_uint8)

            if results['success']:
                frame_features.append(results['features'])

        if not frame_features:
            return np.zeros(len(self.FEATURE_NAMES) * 3)  # 3 stats per feature

        frame_features = np.array(frame_features)

        # Calculate temporal statistics
        feature_stats = []
        for i in range(frame_features.shape[1]):
            feature_stats.extend([
                np.mean(frame_features[:, i]),  # Mean
                np.std(frame_features[:, i]),   # Standard deviation
                np.max(frame_features[:, i])    # Maximum value
            ])

        return np.array(feature_stats)

    def process_dataset(self, input_dir: str, output_dir: str) -> None:
        """
        Process all sequences in a directory to extract features.

        Args:
            input_dir: Directory containing .npy sequence files
            output_dir: Directory to save extracted features
        """
        os.makedirs(output_dir, exist_ok=True)
        sequence_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

        logger.info(f"Processing {len(sequence_files)} sequences from {input_dir}")

        for seq_file in tqdm(sequence_files, desc="Extracting features"):
            try:
                seq_path = os.path.join(input_dir, seq_file)
                sequence = np.load(seq_path)
                features = self.extract_sequence_features(sequence)

                # Save features with same filename in output directory
                output_path = os.path.join(output_dir, seq_file)
                np.save(output_path, features)

            except Exception as e:
                logger.error(f"Error processing {seq_file}: {str(e)}")
                continue

        logger.info(f"Feature extraction complete. Saved to {output_dir}")

def main():
    """Command-line interface for feature extraction."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Driver Drowsiness Feature Extraction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing processed sequences (.npy files)')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save extracted features')
    parser.add_argument('--dataset', type=str, choices=['YawDD', 'NTHU'], required=True,
                      help='Which dataset to process (affects feature selection)')

    args = parser.parse_args()

    # Initialize feature builder
    feature_builder = DrowsinessFeatureBuilder()

    # Process the dataset
    feature_builder.process_dataset(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main()
