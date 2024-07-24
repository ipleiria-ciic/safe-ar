import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import argparse
import importlib
import os
import logging

import src.seg_yolov8
from src.safear_service import SafeARService

# Reload the modules
importlib.reload(src.seg_yolov8)
importlib.reload(src.safear_service)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(args):
    safeARservice = SafeARService()

    safeARservice.configure(
        model_number=args.model_number,
        obfuscation_policies=args.obfuscate_policies,
    )

    # Log the policies
    logger.debug(f"Configured with model_number: {args.model_number}")
    logger.debug(f"Obfuscation policies: {args.obfuscate_policies}")

    frame_bytes = safeARservice.process_frame(args.image_base64)

    return frame_bytes

def parse_args():
    arg_parser = argparse.ArgumentParser(description="Obfuscation script")
    arg_parser.add_argument(
        "--model_number",
        type=int,
        default=0,
        help="Choose the number of the model to use. Use '0' for the default model.",
    )

    arg_parser.add_argument(
        "--class_id_list",
        nargs="+",
        type=int,
        help="Specify the list of class IDs to obfuscate. Separate IDs with spaces.",
    )

    arg_parser.add_argument(
        "--obfuscation_type_list",
        nargs="+",
        type=str,
        help="Specify the list of obfuscation types for each class ID. Separate types with spaces.",
    )

    arg_parser.add_argument(
        "--image_base64_file",
        type=str,
        help="Path to the file containing the Base64-encoded image string.",
    )

    arg_parser.add_argument(
        "--version",
        action="version",
        version="Obfuscation script 1.0",
    )

    arg_parser.add_argument(
        "--square",
        type=int,
        default=0,
        help="Size of the square for pixelation effect.",
    )

    arg_parser.add_argument(
        "--sigma",
        type=int,
        default=0,
        help="Sigma value for the blurring effect.",
    )

    args = arg_parser.parse_args()

    if not args.__dict__:
        arg_parser.print_help()
        exit()

    # Check if class_id_list and obfuscation_type_list are provided
    if args.class_id_list and args.obfuscation_type_list:
        args.obfuscate_policies = dict(zip(args.class_id_list, args.obfuscation_type_list))
    else:
        args.obfuscate_policies = {}

    # Print the absolute path of the file
    abs_path = os.path.abspath(args.image_base64_file)
    print(f"Absolute path of the file: {abs_path}")

    # Read the Base64-encoded image string from the file
    with open(args.image_base64_file, "r") as f:
        args.image_base64 = f.read()

    return args


if __name__ == "__main__":
    main_args = parse_args()

    safeAR_frame_bytes = main(main_args)

    # safeAR_image_base64 = base64.b64encode(safeAR_frame_bytes).decode("utf-8")

    #  DEBUG: save the processed frame
    import cupy as cp
    import imageio

    safeAR_frame_array = cp.frombuffer(safeAR_frame_bytes, dtype=cp.uint8)
    safeAR_frame_array = safeAR_frame_array.reshape((640, 640, 3))
    imageio.imwrite("outputs/OUTPUT_2.png", safeAR_frame_array.get())
