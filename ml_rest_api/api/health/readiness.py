"""This module implements the HealthReadiness class."""
from flask_restx import Resource
from ml_rest_api.api.restx import api, FlaskApiReturnType
from ml_rest_api.ml_trained_model.wrapper import trained_model_wrapper


@api.default_namespace.route("/readiness")
class HealthReadiness(Resource):
    """Implements the /readiness GET method."""

    @staticmethod
    @api.doc(
        responses={
            200: "Success",
            503: "Server Not Ready",
        }
    )
    def get() -> FlaskApiReturnType:
        """
        Returns readiness status.
        """
        _ready = trained_model_wrapper.ready()
        status_code = 200 if _ready else 503
        return {"Ready": _ready}, status_code
