package com.styrs.solarcell.api

import com.google.gson.annotations.SerializedName

/**
 * PredictionResponse — JSON Response Model for /predict Endpoint
 * ================================================================
 *
 * Version : 3.0
 *
 * Maps the JSON response from the Flask API's /predict endpoint to
 * Kotlin data classes.  Gson uses @SerializedName annotations to match
 * the JSON field names (snake_case) to Kotlin property names (camelCase).
 *
 * Example JSON:
 *   {
 *     "success": true,
 *     "predicted_class": "Defective",
 *     "confidence": 0.923,
 *     "probabilities": { "defective": 0.923, "good": 0.077 },
 *     "error": null
 *   }
 *
 * @property success        Whether the prediction completed without errors.
 * @property predictedClass The classification result: "Good" or "Defective".
 * @property confidence     The model's confidence (0.0 to 1.0).
 * @property probabilities  Per-class probability breakdown.
 * @property error          Error message if the prediction failed (null on success).
 */
data class PredictionResponse(
    @SerializedName("success")
    val success: Boolean,

    @SerializedName("predicted_class")
    val predictedClass: String?,

    @SerializedName("confidence")
    val confidence: Double?,

    @SerializedName("probabilities")
    val probabilities: Probabilities?,

    @SerializedName("error")
    val error: String?
)

/**
 * Probabilities — Per-class probability breakdown.
 *
 * @property defective Probability that the cell is defective (0.0 to 1.0).
 * @property good      Probability that the cell is good (0.0 to 1.0).
 *
 * Note: defective + good should approximately equal 1.0 (softmax output).
 */
data class Probabilities(
    @SerializedName("defective")
    val defective: Double,

    @SerializedName("good")
    val good: Double
)

/**
 * HealthResponse — JSON Response Model for /health Endpoint
 * ===========================================================
 *
 * Maps the JSON response from the Flask API's /health endpoint.
 *
 * Example JSON:
 *   { "status": "healthy", "model_loaded": true }
 *
 * @property status      Server status string (e.g. "healthy").
 * @property modelLoaded Whether the AI model is loaded and ready for inference.
 */
data class HealthResponse(
    @SerializedName("status")
    val status: String,

    @SerializedName("model_loaded")
    val modelLoaded: Boolean
)
