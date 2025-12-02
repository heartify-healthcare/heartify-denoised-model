"""ECG Denoising endpoint controller"""
from flask import Blueprint, request, jsonify, current_app
from app.api_keys.auth import api_key_required
from app.denoising.ecg_model import ECGModel
import numpy as np


denoising_bp = Blueprint('denoising', __name__)


@denoising_bp.route('', methods=['POST'])
@api_key_required
def denoise_ecg():
    """
    Perform ECG denoising using the AttentionUNet model

    Requires: x-api-key header with valid API key

    Body:
        {
            "ecg_signal": [array of 1300 float values for 130Hz 1-lead ECG (10 seconds)]
        }

    Returns:
        200: JSON with denoised signal and model version
        400: Invalid request
        401: Invalid or missing API key
        500: Model inference error
    """
    try:
        data = request.get_json()

        # Validate request body
        if not data or 'ecg_signal' not in data:
            return jsonify({
                "error": "Missing required field: ecg_signal",
                "expected_format": {
                    "ecg_signal": "array of 1300 float values (10 seconds @ 130Hz)"
                }
            }), 400

        ecg_signal = data['ecg_signal']

        # Validate ECG signal format
        if not isinstance(ecg_signal, list):
            return jsonify({"error": "ecg_signal must be an array"}), 400

        if len(ecg_signal) != 1300:
            return jsonify({
                "error": f"ecg_signal must have exactly 1300 values (got {len(ecg_signal)})",
                "note": "This model expects 130Hz sampling rate with 10-second duration (1300 samples)"
            }), 400

        # Convert to numpy array
        try:
            ecg_array = np.array(ecg_signal, dtype=np.float32)
        except Exception as e:
            return jsonify({"error": f"Invalid ecg_signal format: {str(e)}"}), 400

        # Get model instance and perform denoising
        model = ECGModel()
        denoised_signal = model.denoise(ecg_array)

        # Get model version from config
        model_version = current_app.config.get('MODEL_VERSION', 1)

        # Build response with denoised signal and model version
        response = {
            "modelVersion": model_version,
            "denoised_signal": denoised_signal.tolist()
        }

        return jsonify(response), 200

    except RuntimeError as e:
        # Model not loaded error
        return jsonify({
            "error": "Model initialization error",
            "details": str(e)
        }), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500
