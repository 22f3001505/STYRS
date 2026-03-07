package com.styrs.solarcell.api

import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

/**
 * RetrofitClient — HTTP Client for API Communication (Legacy)
 * ==============================================================
 *
 * Version : 3.0
 *
 * Configures and provides a Retrofit HTTP client for communicating with
 * the Flask REST API server.  In v3.0, the app uses on-device TFLite
 * inference, so this client is kept as a legacy fallback.
 *
 * Configuration:
 *   - Base URL: Local Flask server (http://192.168.0.6:5001/)
 *   - Timeouts: 30s connect, 60s read/write
 *   - Logging: Full request/response body logging (for debugging)
 *   - JSON parsing: Gson converter
 *
 * Usage:
 *   val response = RetrofitClient.apiService.predict(imagePart)
 *   val health = RetrofitClient.apiService.healthCheck()
 */
object RetrofitClient {

    // Base URL of the local Flask API server
    // This must match the IP address of the machine running api_server.py
    private const val BASE_URL = "http://192.168.0.6:5001/"

    // Note: The HF Space (https://jayaram060504-styrs-solar-inspector.hf.space)
    // runs the Streamlit UI only — it does not expose a REST API.
    // For the Android app, use the local Flask server or on-device TFLite.

    // HTTP logging interceptor — logs request/response details for debugging
    private val loggingInterceptor = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.BODY   // Log full body content
    }

    // OkHttp client with custom timeouts for large image uploads
    private val okHttpClient = OkHttpClient.Builder()
        .addInterceptor(loggingInterceptor)
        .connectTimeout(30, TimeUnit.SECONDS)   // Time to establish connection
        .readTimeout(60, TimeUnit.SECONDS)      // Time to receive response
        .writeTimeout(60, TimeUnit.SECONDS)     // Time to send request body
        .build()

    // Retrofit instance configured with the base URL and Gson JSON parser
    private val retrofit = Retrofit.Builder()
        .baseUrl(BASE_URL)
        .client(okHttpClient)
        .addConverterFactory(GsonConverterFactory.create())
        .build()

    /** Default API service instance using the hardcoded BASE_URL */
    val apiService: ApiService = retrofit.create(ApiService::class.java)

    /**
     * Create a new API service instance with a custom base URL.
     *
     * This is useful when the user changes the server URL in Settings,
     * allowing the app to connect to a different server without restarting.
     *
     * @param baseUrl The new server URL (e.g. "http://192.168.1.100:5001/")
     * @return A new ApiService instance configured for the specified URL.
     */
    fun createService(baseUrl: String): ApiService {
        val customRetrofit = Retrofit.Builder()
            .baseUrl(baseUrl)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
        return customRetrofit.create(ApiService::class.java)
    }
}
