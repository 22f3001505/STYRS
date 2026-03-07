package com.styrs.solarcell.api

import okhttp3.MultipartBody
import retrofit2.Response
import retrofit2.http.*

/**
 * ApiService — Retrofit Interface for the Flask REST API (Legacy)
 * =================================================================
 *
 * Version : 3.0
 *
 * Defines the HTTP endpoints for communicating with the Flask API server
 * (api_server.py).  In v3.0, on-device TFLite inference is used instead,
 * so this interface is kept as a legacy fallback for potential future use.
 *
 * Endpoints:
 *   POST /predict  — Upload an image for classification
 *   GET  /health   — Check if the server is running and model is loaded
 */
interface ApiService {

    /**
     * Upload a solar cell image and receive a classification prediction.
     *
     * @param image The image file as a multipart form-data body part.
     * @return Response containing the predicted class, confidence, and probabilities.
     */
    @Multipart
    @POST("predict")
    suspend fun predict(
        @Part image: MultipartBody.Part
    ): Response<PredictionResponse>

    /**
     * Check the server's health status and whether the model is loaded.
     *
     * @return Response containing the server status and model loading state.
     */
    @GET("health")
    suspend fun healthCheck(): Response<HealthResponse>
}
