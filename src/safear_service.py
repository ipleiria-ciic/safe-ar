import base64
import os
from io import BytesIO

import cupy as cp
import imageio
import yaml

from src.obfuscator import ImageObfuscator
from src.seg_yolov8 import Yolov8seg

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

    def process_frame(self, image_base64: str) -> bytes:
        """
        Process a frame by detecting objects and applying obfuscation.
        Args:
            image_base64: str representation of an image, encoded in Base64 format

        Returns:
            safe_frame_bytes: the processed frame, encoded in bytes
        """
        try:
            # Decode the Base64 image string to bytes
            image_bytes = base64.b64decode(image_base64.encode("utf-8"))
            print(f"Image bytes length: {len(image_bytes)}")
            
            # Create a buffer (`BytesIO` object) from the image bytes
            buffer = BytesIO(image_bytes)

            # Read the image from the buffer using imageio
            img_array = imageio.v2.imread(buffer)
            # Convert the Numpy array to a cuPY array
            frame = cp.asarray(img_array)

            # DEBUG: Save the original frame
            original_frame_path = "outputs/original_frame.png"
            imageio.imwrite(original_frame_path, frame.get())

            # Perform inference
            boxes, masks = self.model(frame)
            print(f"Model output boxes: {boxes}")
            print(f"Model output masks: {masks}")

            # Check if the model returned any boxes or masks
            if len(boxes) == 0 or len(masks) == 0:
                print("No objects detected, returning the original frame.")
                safe_frame = frame
            else:
                safe_frame = self.obfuscator.obfuscate(
                    image=frame, masks=masks, class_ids=[int(box[5]) for box in boxes]
                )

            safe_frame = safe_frame.astype(cp.uint8)
            
            # DEBUG: Save the obfuscated frame
            obfuscated_frame_path = "outputs/obfuscated_frame.png"
            imageio.imwrite(obfuscated_frame_path, safe_frame.get())

            # Convert the processed frame to bytes
            safe_frame_bytes = safe_frame.tobytes()

            return safe_frame_bytes

        except Exception as e:
            print(f"Error processing image: {e}")
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
        # Get the current directory
        current_directory = os.path.dirname(os.path.abspath(__file__))
        # Get the parent directory
        parent_directory = os.path.dirname(current_directory)
        # Construct the path to the config.yml file
        config_file_path = os.path.join(parent_directory, "config.yml")

        with open(file=config_file_path, mode="r", encoding="utf-8") as file:
            config_yml = yaml.safe_load(file)
        return config_yml

    @staticmethod
    def list_models() -> list:
        config_yml = SafeARService.load_config()
        return list(config_yml["models"].keys())
