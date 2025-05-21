"""This module implements the SegmentationBackgroundRemoval class."""
import io
import os
import tempfile
from flask import request, send_file
from flask_restx import Resource, reqparse
from PIL import Image
from werkzeug.datastructures import FileStorage
from ml_rest_api.api.restx import api
from ml_rest_api.ml_trained_model.ml_trained_model import init as model_init
from rembg import remove

# Initialize the ML model on import
model_init()

# Request parser for file upload
upload_parser = reqparse.RequestParser()
upload_parser.add_argument(
    'file',
    location='files',
    type=FileStorage,
    required=True,
    help='Image file to remove background from'
)

ns = api.namespace(
    "segmentation",
    description="Segmentation methods to detect objects in images."
)

@ns.route("/background_removal")
class SegmentationBackgroundRemoval(Resource):
    """Implements the /segmentation/background_removal POST method."""

    @ns.expect(upload_parser)
    @ns.doc(
        responses={
            200: "Success",
            400: "Input Validation Error",
            500: "Internal Server Error",
        }
    )
    def post(self):
        """
        Removes the background from the uploaded image and returns the processed image.
        """
        if 'file' not in request.files:
            return {'error': 'No file part in request'}, 400

        file = request.files['file']
        if file.filename == '':
            return {'error': 'No selected file'}, 400

        try:
            img_bytes = file.read()
            input_image = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

            # Remove the background using rembg
            output_image = remove(input_image)

            # Save output image to a temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            temp_file.close()  # Close so PIL can write to it
            output_image.save(temp_file.name, format="PNG")

            # Generate a meaningful download filename
            original_filename = file.filename
            base_filename = os.path.splitext(original_filename)[0]
            download_filename = f"bg_removed_{base_filename}.png"

            # Send the file as an attachment
            return send_file(
                temp_file.name,
                mimetype='image/png',
                as_attachment=True,
                download_name=download_filename
            )
        except Exception as e:
            # Handle unexpected errors gracefully
            return {'error': f'Failed to process image: {str(e)}'}, 500
