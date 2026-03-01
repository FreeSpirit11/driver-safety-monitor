import os
import numpy as np
import cv2
from tqdm import tqdm
import logging
from typing import Tuple
from feature_extractor import extract_features_from_frame

# Logging setup
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class AttentivenessFeatureBuilder:
    """
    Feature extraction pipeline for driver attentiveness.
    Extracts gaze and head orientation vectors and computes temporal statistics.
    """

    def __init__(self):
        self.feature_dim = 4  # gaze_x, gaze_y, head_x, head_y

    def extract_frame_features(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Extract gaze and head direction features from a single frame.

        Args:
            frame: Input image (BGR)

        Returns:
            A tuple (features, success) where:
                - features: np.ndarray of shape (4,)
                - success: True if face detected, else False
        """
        features = extract_features_from_frame(frame)
        return (features, features is not None)

    def extract_sequence_features(self, sequence: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from a sequence of frames.

        Args:
            sequence: np.ndarray of shape (seq_len, H, W, C)

        Returns:
            np.ndarray of shape (n_features,) — aggregated features
        """
        frame_features = []

        for frame in sequence:
            frame_uint8 = (frame * 255).astype(np.uint8)
            features, success = self.extract_frame_features(frame_uint8)
            if success:
                frame_features.append(features)

        if not frame_features:
            return np.zeros(self.feature_dim * 3)  # mean, std, max per feature

        frame_features = np.array(frame_features)

        stats = []
        for i in range(frame_features.shape[1]):
            stats.extend([
                np.mean(frame_features[:, i]),
                np.std(frame_features[:, i]),
                np.max(frame_features[:, i])
            ])

        return np.array(stats)

    def process_dataset(self, input_dir: str, output_dir: str) -> None:
        """
        Process .npy sequences from input_dir, extract features, and save them to output_dir.

        Args:
            input_dir: Directory with input .npy sequences
            output_dir: Directory to save extracted .npy features
        """
        os.makedirs(output_dir, exist_ok=True)
        sequence_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

        logger.info(f"Processing {len(sequence_files)} sequences from {input_dir}")

        for filename in tqdm(sequence_files, desc="Extracting attentiveness features"):
            try:
                sequence = np.load(os.path.join(input_dir, filename))
                features = self.extract_sequence_features(sequence)
                np.save(os.path.join(output_dir, filename), features)
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")

        logger.info(f"Saved extracted features to {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Driver Attentiveness Feature Extraction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing .npy video sequences')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save extracted features')

    args = parser.parse_args()

    feature_builder = AttentivenessFeatureBuilder()
    feature_builder.process_dataset(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
