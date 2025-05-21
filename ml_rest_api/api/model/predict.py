"""This module implements the ModelPredict class."""
import uuid
import os
from typing import Dict
from flask import request
from flask_restx import Resource, reqparse
from ml_rest_api.api.restx import api, FlaskApiReturnType, MLRestAPINotReadyException
from ml_rest_api.ml_trained_model.wrapper import trained_model_wrapper
from ml_rest_api.ml_trained_model.ml_trained_model import full_path
from werkzeug.datastructures import FileStorage

# Request parser for file upload and classifiers
upload_parser = reqparse.RequestParser()
upload_parser.add_argument(
    'file',
    location='files',
    type=FileStorage,
    required=True,
    help="Image file to classify"
)
upload_parser.add_argument(
    'classifiers',
    required=True,
    action='append',
    help="List of classifiers, e.g., ['cardboard', 'glass', 'metal', 'paper','plastic', 'trash']"
)

ns = api.namespace(
    "model",
    description="Methods supported by our ML model",
    validate=bool(trained_model_wrapper.sample()),
)

@ns.route("/predict")
class ModelPredict(Resource):
    """Implements the /model/predict POST method."""

    @api.expect(upload_parser)
    @api.doc(
        responses={
            200: "Success",
            400: "Input Validation Error",
            500: "Internal Server Error",
            503: "Server Not Ready",
        }
    )
    def post(self):
        """
        Returns a prediction using the model.
        """
        # Uncomment this to enforce readiness check
        # if not trained_model_wrapper.ready():
        #     raise MLRestAPINotReadyException()

        if 'file' not in request.files:
            return {'error': 'No file part in request'}, 400

        file = request.files['file']
        if file.filename == '':
            return {'error': 'No selected file'}, 400

        # Save uploaded file temporarily with a unique UUID filename
        filename = f"{uuid.uuid4()}.jpg"
        temp_file = full_path(f"temp/{filename}")

        img_bytes = file.read()
        with open(temp_file, mode="wb") as jpg_file:
            jpg_file.write(img_bytes)

        try:
            args = upload_parser.parse_args()
            model_dict: Dict = {
                "image": temp_file,
                "classifiers": args['classifiers']
            }
            prediction_result = trained_model_wrapper.run(model_dict)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)

        return prediction_result, 200
