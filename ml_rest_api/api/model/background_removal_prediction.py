"""This module implements the SegmentationMasking and ModelPredict classes."""
from flask import request, send_file, jsonify
from flask_restx import Resource, reqparse
from PIL import Image
import io
import os
import tempfile
from typing import Dict
from werkzeug.datastructures import FileStorage
from ml_rest_api.api.restx import api
from ml_rest_api.ml_trained_model.ml_trained_model import full_path
from ml_rest_api.ml_trained_model.wrapper import trained_model_wrapper
from rembg import remove

# Define the request parser for file uploads and classifiers list
upload_parser = reqparse.RequestParser()
upload_parser.add_argument(
    'file',
    location='files',
    type=FileStorage,
    required=True,
    help="Image file to process"
)
upload_parser.add_argument(
    'classifiers',
    required=True,
    action='append',
    help="List of classifiers e.g. ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper','plastic','shoes','trash']"
)

ns = api.namespace(
    "model",
    description="Methods supported by our ML model",
    validate=bool(trained_model_wrapper.sample()),
)

@ns.route("/background_removal_predict")
class BackgroundRemovalPredict(Resource):
    """Implements the /model/background_removal_predict POST method."""

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
        Removes the background of the image and returns a prediction using the model.
        """
        if 'file' not in request.files:
            return {'error': 'No file part in request'}, 400

        file = request.files['file']
        if file.filename == '':
            return {'error': 'No selected file'}, 400

        try:
            img_bytes = file.read()
            input_image = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        except Exception as e:
            return {'error': f'Failed to process image: {str(e)}'}, 400

        # Remove background using rembg
        output_image = remove(input_image)

        # Save the result image temporarily in system temp directory
        result_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        try:
            output_image.save(result_file.name, format="PNG")
        finally:
            result_file.close()

        # Construct download filename with PNG extension
        original_filename = file.filename
        base_filename = os.path.splitext(original_filename)[0]
        download_filename = f"bg_removed_{base_filename}.png"

        # Save the processed image to a known path for prediction
        temp_file = full_path("temp/upload.png")
        output_image.save(temp_file, format="PNG")

        # Parse classifiers argument
        args = upload_parser.parse_args()

        model_dict: Dict = {
            "image": temp_file,
            "classifiers": args['classifiers']
        }

        # Perform prediction
        prediction_result = trained_model_wrapper.run(model_dict)

        return jsonify({
            'prediction': prediction_result,
            'processed_image_url': request.host_url + 'api/model/download_processed_image/' + os.path.basename(result_file.name)
        })

@ns.route("/download_processed_image/<filename>")
class DownloadProcessedImage(Resource):
    """Serves the processed image for download."""

    def get(self, filename):
        """
        Returns the processed image file.
        """
        file_path = os.path.join(tempfile.gettempdir(), filename)
        if os.path.exists(file_path):
            return send_file(
                file_path,
                mimetype='image/png',
                as_attachment=True,
                download_name=filename
            )
        else:
            return {'error': 'File not found'}, 404
