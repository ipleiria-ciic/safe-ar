"""
SafeAR - Image Obfuscation Service

This Flask application provides an image obfuscation service using the SafeARService.
It exposes several endpoints for different functionalities:

- `/obfuscate`: Accepts POST requests with image data, processes the image,
 and returns the obfuscated image data in base64 format.
- `/status`: Returns the status of the service.
- `/`: Serves the homepage, which is an `index.html` file.
- `/static/<path:path>`: Serves static files from the "static" directory.

Usage:
    To use the SafeAR service, send a POST request with image data to `/obfuscate`.
    The server will respond with obfuscated image data in base64 format.
    Decode the image data to obtain the processed image.

    Example:
    curl -X POST -H "Content-Type: application/octet-stream"
    --data "$(base64 -i image.jpg)" http://localhost:5000/obfuscate

Note:
    - kill the server: curl -X POST http://localhost:5000/shutdown or CTRL+C
"""

import sys
import os

# Add the project directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import base64
import logging
from flask import Flask, request, render_template, send_from_directory, jsonify
from src.safear_service import SafeARService

# Initialize the Flask app
app = Flask(__name__)

# Set logging level
app.logger.setLevel(logging.DEBUG)  # Change to DEBUG for detailed logs

# Configure logging handler (e.g., console)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
app.logger.addHandler(handler)

# Disable caching of static files
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.config["STATIC_FOLDER"] = "templates"


# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@app.route("/obfuscate", methods=["POST"])
def safeAR_service():
    try:
        data = request.get_json()
        model_number = data.get('model_number', 0)
        class_id_list = data.get('class_id_list')
        obfuscation_type_list = data.get('obfuscation_type_list')

        # Log the incoming request data
        logging.debug(f"Received request data: {data}")

        if data is None or "img" not in data:
            logger.error("No valid request body or 'img' missing in JSON")
            return jsonify({"error": "No valid request body, json missing!"}), 400
        
        img_data = data["img"]
        logger.info(f"Received base64 image data of length: {len(img_data)}")
        
        img_data += "=" * ((4 - len(img_data) % 4) % 4)
        
        safe_ar_service = SafeARService()
        obfuscation_policies = data.get("obfuscation_type_list")
        obfuscation_policies = dict(zip(class_id_list, obfuscation_type_list))
        logging.info(f"Obfuscation policies: {obfuscation_policies}")
        
        safe_ar_service.configure(model_number=0, obfuscation_policies=obfuscation_policies)
        
        processed_frame_bytes = safe_ar_service.process_frame(img_data)
        
        if not processed_frame_bytes:
            logger.error("Processed frame is empty")
            return jsonify({"error": "Processed frame is empty"}), 500
        
        safeAR_image_base64 = base64.b64encode(processed_frame_bytes).decode("utf-8")
        logger.info(f"Returning base64 image of length: {len(safeAR_image_base64)}")
        logger.debug(f"Returning base64 image: {safeAR_image_base64[:100]}")
        
        return jsonify({"img": safeAR_image_base64})
    except Exception as e:
        logger.exception(f"Error processing image: {e}")
        return jsonify({"error": "Failed to process image"}), 500


@app.route("/status")
def status():
    return jsonify({"status": "SafeAR Obfuscation Service is running"})


@app.route("/health")
def health():
    return jsonify({"status": "healthy"})


@app.route("/")
def index():
    # Render the index.html file as the homepage
    return render_template("index.html")


@app.route("/static/<path:path>")
def send_static(path):
    # Serve static files from the "static" folder
    return send_from_directory("static", path)


def shutdown_server():
    func = request.environ.get("werkzeug.server.shutdown")
    if func is None:
        raise RuntimeError("Not running with the Werkzeug Server")
    func()


@app.route("/shutdown", methods=["POST"])
def shutdown():
    shutdown_server()
    return "Server shutting down..."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

