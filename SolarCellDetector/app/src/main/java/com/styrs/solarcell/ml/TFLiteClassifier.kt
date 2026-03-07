package com.styrs.solarcell.ml

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * TFLiteClassifier — On-Device Solar Cell Defect Classifier
 * ==========================================================
 *
 * Version : 3.0
 * Date    : February 2026
 *
 * This class wraps a TensorFlow Lite model to perform real-time solar cell
 * defect classification directly on the Android device — no internet or
 * server connection required.
 *
 * How it works:
 *   1. The pre-trained EfficientNetB3 model was converted from Keras (.keras)
 *      to TFLite (.tflite) format with Float16 quantisation, reducing the
 *      model size from 153 MB to 25 MB.
 *   2. At app startup, the model is memory-mapped from the APK's assets folder
 *      (zero-copy loading for efficiency).
 *   3. When classify() is called, the input bitmap is:
 *      a) Resized to 300×300 pixels
 *      b) Normalised from [0, 255] to [0.0, 1.0]
 *      c) Packed into a ByteBuffer in RGB pixel order
 *   4. The TFLite interpreter runs inference using 4 CPU threads.
 *   5. The output is a Float[2] array: [P(Defective), P(Good)].
 *
 * Usage:
 *   val classifier = TFLiteClassifier(context)
 *   classifier.initialize()
 *   val result = classifier.classify(bitmap)
 *   // result.predictedClass → "Good" or "Defective"
 *   // result.confidence → 0.0 to 1.0
 *   classifier.close()
 */
class TFLiteClassifier(private val context: Context) {

    companion object {
        /** Name of the TFLite model file in the assets/ folder */
        private const val MODEL_FILE = "solar_cell_model.tflite"

        /** The model expects 300×300 RGB images as input */
        private const val IMAGE_SIZE = 300

        /** Binary classification: Defective vs Good */
        private const val NUM_CLASSES = 2

        /** Class labels — order must match the model's training data directory order */
        private val CLASS_LABELS = arrayOf("Defective", "Good")
    }

    /** TFLite interpreter instance — null until initialize() is called */
    private var interpreter: Interpreter? = null

    /**
     * Data class to hold the classification result.
     *
     * @property predictedClass  The human-readable class name ("Good" or "Defective")
     * @property confidence      The probability of the predicted class (0.0 to 1.0)
     * @property probDefective   Raw probability that the cell is defective
     * @property probGood        Raw probability that the cell is good
     * @property success         Whether classification completed without errors
     * @property error           Error message if classification failed
     */
    data class ClassificationResult(
        val predictedClass: String,
        val confidence: Double,
        val probDefective: Double,
        val probGood: Double,
        val success: Boolean = true,
        val error: String? = null
    )

    /**
     * Load the TFLite model from the APK's assets folder.
     *
     * The model file is memory-mapped (not copied into RAM), which is both
     * faster and more memory-efficient.  We use 4 threads for inference
     * to take advantage of modern multi-core mobile CPUs.
     *
     * @return true if the model was loaded successfully, false otherwise.
     */
    fun initialize(): Boolean {
        return try {
            val model = loadModelFile()
            val options = Interpreter.Options().apply {
                setNumThreads(4)   // Use 4 CPU threads for parallel inference
            }
            interpreter = Interpreter(model, options)
            true
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }

    /**
     * Classify a bitmap image of a solar cell.
     *
     * This is the main entry point for inference.  It handles the full
     * pipeline: resize → normalise → run model → parse output.
     *
     * @param bitmap The input image (any size — it will be resized internally).
     * @return A ClassificationResult with the prediction details.
     */
    fun classify(bitmap: Bitmap): ClassificationResult {
        val interp = interpreter ?: return ClassificationResult(
            predictedClass = "Unknown",
            confidence = 0.0,
            probDefective = 0.0,
            probGood = 0.0,
            success = false,
            error = "Model not initialized — call initialize() first"
        )

        try {
            // Step 1: Resize the input image to the model's expected dimensions
            val resized = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)

            // Step 2: Convert the bitmap to a normalised ByteBuffer
            val inputBuffer = preprocessImage(resized)

            // Step 3: Prepare the output array — shape [1][2] for batch=1, classes=2
            val outputArray = Array(1) { FloatArray(NUM_CLASSES) }

            // Step 4: Run inference through the TFLite model
            interp.run(inputBuffer, outputArray)

            // Step 5: Parse the softmax output probabilities
            val probabilities = outputArray[0]
            val probDefective = probabilities[0].toDouble()
            val probGood = probabilities[1].toDouble()

            // The predicted class is whichever has the higher probability
            val predictedIndex = if (probGood > probDefective) 1 else 0
            val predictedClass = CLASS_LABELS[predictedIndex]
            val confidence = maxOf(probDefective, probGood)

            return ClassificationResult(
                predictedClass = predictedClass,
                confidence = confidence,
                probDefective = probDefective,
                probGood = probGood
            )
        } catch (e: Exception) {
            e.printStackTrace()
            return ClassificationResult(
                predictedClass = "Unknown",
                confidence = 0.0,
                probDefective = 0.0,
                probGood = 0.0,
                success = false,
                error = e.message
            )
        }
    }

    /**
     * Convert a bitmap to a normalised ByteBuffer for model input.
     *
     * The model expects a Float32 tensor of shape [1, 300, 300, 3] with
     * pixel values in the range [0.0, 1.0].  We extract each pixel's
     * R, G, B channels and divide by 255.
     *
     * Memory layout: R0 G0 B0 R1 G1 B1 R2 G2 B2 ...  (interleaved RGB)
     *
     * @param bitmap A 300×300 Bitmap (already resized).
     * @return ByteBuffer containing the normalised pixel data.
     */
    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        // Allocate a direct buffer: 4 bytes per float × 300 × 300 × 3 channels
        val buffer = ByteBuffer.allocateDirect(4 * IMAGE_SIZE * IMAGE_SIZE * 3)
        buffer.order(ByteOrder.nativeOrder())   // Use the device's native byte order

        // Read all pixels into an IntArray (each pixel is packed as ARGB)
        val pixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)
        bitmap.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)

        // Extract R, G, B channels from each pixel and normalise to [0, 1]
        for (pixel in pixels) {
            val red   = ((pixel shr 16) and 0xFF) / 255.0f   // Bits 16-23 = Red
            val green = ((pixel shr 8) and 0xFF) / 255.0f    // Bits 8-15  = Green
            val blue  = (pixel and 0xFF) / 255.0f            // Bits 0-7   = Blue

            buffer.putFloat(red)
            buffer.putFloat(green)
            buffer.putFloat(blue)
        }

        buffer.rewind()   // Reset position to 0 so the interpreter reads from the start
        return buffer
    }

    /**
     * Load the TFLite model file from the APK's assets as a memory-mapped buffer.
     *
     * Memory-mapping avoids copying the entire 25 MB model into the Java heap.
     * Instead, the OS maps the file directly into the process's address space,
     * which is both faster and uses less RAM.
     *
     * @return MappedByteBuffer pointing to the model file.
     */
    private fun loadModelFile(): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(MODEL_FILE)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * Release the TFLite interpreter resources.
     *
     * Call this in the Activity's onDestroy() to free native memory
     * used by the interpreter.
     */
    fun close() {
        interpreter?.close()
        interpreter = null
    }
}
