import base64
import os
from io import BytesIO

import cupy as cp
import imageio
import yaml
import logging

from src.obfuscator import ImageObfuscator
from src.seg_yolov8 import Yolov8seg

logger = logging.getLogger(__name__)

class SafeARService:
    def __init__(self):
        self.obfuscator = None
        self.model = None
        self.obfuscation_policies = {}

    def configure(self, model_number: int, obfuscation_policies: dict):
        config_yml = self.load_config()
        model_name = list(config_yml["models"].keys())[model_number]
        model_path = config_yml["models"][model_name]["model_path"]

        self.model = Yolov8seg(model_path=model_path)
        self.obfuscation_policies = obfuscation_policies
        self.obfuscator = ImageObfuscator(policies=self.obfuscation_policies)

    def process_frame(self, image_base64: str):
        try:
            # Decode the base64 image data
            image_bytes = base64.b64decode(image_base64.encode("utf-8"))
            logger.info(f"Decoded image bytes length: {len(image_bytes)}")

            # Read the image data into an array
            buffer = BytesIO(image_bytes)
            img_array = imageio.v2.imread(buffer)
            logger.info(f"Read image array shape: {img_array.shape}")

            frame = cp.asarray(img_array)

            # Process the image using the model
            boxes, masks = self.model(frame)
            logger.info(f"Model output: {len(boxes)} boxes, {len(masks)} masks")

            if len(boxes) == 0 or len(masks) == 0:
                logger.warning("No objects detected.")
                return []

            detected_objects = []
            for i, box in enumerate(boxes):
                detected_objects.append({
                    'class_id': int(box[5]),
                    'bbox': [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                    'mask': cp.asnumpy(masks[i]).tolist()  # Convert mask to a list of lists for JSON serialization
                })

            return detected_objects
        except Exception as e:
            logger.exception(f"Error processing image: {e}")
            return None

    @staticmethod
    def read_base64_image(file_path):
        with open(file_path, "r") as f:
            image_base64 = f.read()
        image_data = base64.b64decode(image_base64)
        return image_data

    @staticmethod
    def save_processed_frame(frame_bytes, output_path):
        frame_array = cp.frombuffer(frame_bytes, dtype=cp.uint8)
        if len(frame_array) != 640 * 640 * 3:
            raise ValueError("Incorrect size of frame data")
        frame_array = frame_array.reshape((640, 640, 3))
        # Convert cupy array to numpy array
        frame_array = cp.asnumpy(frame_array)
        imageio.imwrite(output_path, frame_array)

    @staticmethod
    def load_config() -> dict:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        parent_directory = os.path.dirname(current_directory)
        config_file_path = os.path.join(parent_directory, "config.yml")

        with open(file=config_file_path, mode="r", encoding="utf-8") as file:
            config_yml = yaml.safe_load(file)
        return config_yml

    @staticmethod
    def list_models() -> list:
        config_yml = SafeARService.load_config()
        return list(config_yml["models"].keys())
