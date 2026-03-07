package com.styrs.solarcell.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.view.isVisible
import androidx.lifecycle.lifecycleScope
import coil.load
import com.styrs.solarcell.R
import com.styrs.solarcell.data.AppSettings
import com.styrs.solarcell.data.ScanHistoryManager
import com.styrs.solarcell.data.ScanRecord
import com.styrs.solarcell.databinding.ActivityMainBinding
import com.styrs.solarcell.ml.TFLiteClassifier
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

/**
 * MainActivity — Central Hub of the STYRS Android App
 * =====================================================
 *
 * Version : 3.0
 *
 * This activity is the main screen where users can:
 *   1. Capture a solar cell photo using the device camera
 *   2. Select an existing image from the gallery
 *   3. Preview the selected image
 *   4. Tap "ANALYZE" to run on-device AI classification
 *   5. Navigate to History or Settings screens
 *
 * The classification is performed entirely on-device using TFLite,
 * so no internet connection or remote server is required.
 *
 * Flow:
 *   SplashActivity → [this] MainActivity → ResultActivity
 *                          ↓
 *                   HistoryActivity / SettingsActivity
 */
class MainActivity : AppCompatActivity() {

    /** View binding — provides type-safe access to all UI elements */
    private lateinit var binding: ActivityMainBinding

    /** URI of the currently selected or captured image */
    private var selectedImageUri: Uri? = null

    /** Absolute path to the last photo taken by the camera */
    private var currentPhotoPath: String? = null

    /** On-device TFLite classifier for solar cell defect detection */
    private lateinit var classifier: TFLiteClassifier

    // ─────────────────────────────────────────────
    // PERMISSION & RESULT LAUNCHERS
    // ─────────────────────────────────────────────
    // Android requires runtime permissions for camera and storage access.
    // We use the modern Activity Result API (registerForActivityResult)
    // instead of the deprecated onActivityResult / onRequestPermissionsResult.

    /** Request camera permission → launch camera if granted */
    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            launchCamera()
        } else {
            Toast.makeText(this, "Camera permission is required", Toast.LENGTH_SHORT).show()
        }
    }

    /** Request storage permission → launch gallery if granted */
    private val galleryPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            launchGallery()
        } else {
            Toast.makeText(this, "Storage permission is required", Toast.LENGTH_SHORT).show()
        }
    }

    /** Handle camera capture result — the photo is saved to currentPhotoPath */
    private val takePictureLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success ->
        if (success) {
            currentPhotoPath?.let { path ->
                selectedImageUri = Uri.fromFile(File(path))
                displaySelectedImage()
                // Auto-analyze if the user has enabled this option in Settings
                if (AppSettings.isAutoAnalyzeEnabled(this)) {
                    analyzeImage()
                }
            }
        }
    }

    /** Handle gallery selection result — the chosen image URI is returned */
    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            selectedImageUri = it
            displaySelectedImage()
            if (AppSettings.isAutoAnalyzeEnabled(this)) {
                analyzeImage()
            }
        }
    }

    // ─────────────────────────────────────────────
    // LIFECYCLE
    // ─────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Initialise the on-device TFLite classifier
        classifier = TFLiteClassifier(this)
        initializeModel()

        // Wire up all button click listeners
        setupClickListeners()
    }

    override fun onDestroy() {
        super.onDestroy()
        // Release native memory used by the TFLite interpreter
        classifier.close()
    }

    // ─────────────────────────────────────────────
    // MODEL INITIALISATION
    // ─────────────────────────────────────────────

    /**
     * Load the TFLite model on a background thread.
     *
     * Model loading is done asynchronously to avoid blocking the UI thread.
     * After loading, we update the status indicator in the header to show
     * "● Model Ready" (green) or "● Model Error" (red).
     */
    private fun initializeModel() {
        lifecycleScope.launch(Dispatchers.IO) {
            val loaded = classifier.initialize()
            withContext(Dispatchers.Main) {
                binding.serverStatus.isVisible = true
                if (loaded) {
                    binding.serverStatus.text = "● Model Ready"
                    binding.serverStatus.setTextColor(
                        ContextCompat.getColor(this@MainActivity, R.color.status_healthy)
                    )
                } else {
                    binding.serverStatus.text = "● Model Error"
                    binding.serverStatus.setTextColor(
                        ContextCompat.getColor(this@MainActivity, R.color.status_critical)
                    )
                }
            }
        }
    }

    // ─────────────────────────────────────────────
    // CLICK LISTENERS
    // ─────────────────────────────────────────────

    /**
     * Wire up all interactive UI elements.
     */
    private fun setupClickListeners() {
        // Camera button — capture a new photo
        binding.btnCamera.setOnClickListener {
            checkCameraPermission()
        }

        // Gallery button — pick an existing image
        binding.btnGallery.setOnClickListener {
            checkGalleryPermission()
        }

        // Tapping the image preview card also opens the gallery
        binding.cardImagePreview.setOnClickListener {
            checkGalleryPermission()
        }

        // Analyze button — run on-device AI classification
        binding.btnAnalyze.setOnClickListener {
            if (selectedImageUri != null) {
                analyzeImage()
            } else {
                Toast.makeText(this, getString(R.string.error_no_image), Toast.LENGTH_SHORT).show()
            }
        }

        // History button — navigate to scan history screen
        binding.btnHistory.setOnClickListener {
            startActivity(Intent(this, HistoryActivity::class.java))
            overridePendingTransition(android.R.anim.slide_in_left, android.R.anim.slide_out_right)
        }

        // Settings button — navigate to app settings screen
        binding.btnSettings.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
            overridePendingTransition(android.R.anim.slide_in_left, android.R.anim.slide_out_right)
        }
    }

    // ─────────────────────────────────────────────
    // PERMISSIONS
    // ─────────────────────────────────────────────

    /**
     * Check if the camera permission is already granted; if not, request it.
     */
    private fun checkCameraPermission() {
        when {
            ContextCompat.checkSelfPermission(
                this, Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> {
                launchCamera()
            }
            else -> {
                cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }

    /**
     * Check if the storage/media permission is already granted; if not, request it.
     *
     * On Android 13+ (Tiramisu), we request READ_MEDIA_IMAGES instead of
     * READ_EXTERNAL_STORAGE because Google refined the permission model.
     */
    private fun checkGalleryPermission() {
        val permission = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            Manifest.permission.READ_MEDIA_IMAGES
        } else {
            Manifest.permission.READ_EXTERNAL_STORAGE
        }

        when {
            ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED -> {
                launchGallery()
            }
            else -> {
                galleryPermissionLauncher.launch(permission)
            }
        }
    }

    // ─────────────────────────────────────────────
    // CAMERA & GALLERY
    // ─────────────────────────────────────────────

    /**
     * Create a temporary file and launch the system camera to capture a photo.
     *
     * We use FileProvider to generate a content:// URI that the camera app
     * can write to.  This avoids exposing raw file:// URIs, which is
     * forbidden on Android 7.0+ (Nougat).
     */
    private fun launchCamera() {
        val photoFile = createImageFile()
        photoFile?.let { file ->
            val photoUri = FileProvider.getUriForFile(
                this,
                "${packageName}.fileprovider",
                file
            )
            currentPhotoPath = file.absolutePath
            takePictureLauncher.launch(photoUri)
        }
    }

    /**
     * Launch the system image picker to select a photo from the gallery.
     */
    private fun launchGallery() {
        pickImageLauncher.launch("image/*")
    }

    /**
     * Create a uniquely named temporary file for storing a camera capture.
     *
     * The filename includes a timestamp to ensure uniqueness and prevent
     * overwrites if the user takes multiple photos in the same session.
     */
    private fun createImageFile(): File? {
        return try {
            val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            val storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
            File.createTempFile("SOLAR_${timeStamp}_", ".jpg", storageDir).also {
                currentPhotoPath = it.absolutePath
            }
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    // ─────────────────────────────────────────────
    // IMAGE DISPLAY
    // ─────────────────────────────────────────────

    /**
     * Show the selected image in the preview card.
     *
     * Hides the placeholder ("Tap to select image") and shows the actual
     * image using the Coil library for efficient async loading with
     * a crossfade transition.
     */
    private fun displaySelectedImage() {
        binding.placeholderContainer.isVisible = false
        binding.imagePreview.isVisible = true
        binding.imagePreview.load(selectedImageUri) {
            crossfade(true)   // Smooth fade-in animation
        }
    }

    // ─────────────────────────────────────────────
    // ON-DEVICE AI ANALYSIS
    // ─────────────────────────────────────────────

    /**
     * Run the TFLite model on the selected image and navigate to results.
     *
     * Pipeline:
     *   1. Decode the image URI into a Bitmap (on background thread)
     *   2. Pass the bitmap to TFLiteClassifier.classify()
     *   3. Save the result to scan history (if enabled in settings)
     *   4. Navigate to ResultActivity with the prediction data
     *
     * All heavy work (decoding, inference) runs on Dispatchers.IO to
     * keep the UI responsive.  A loading overlay is shown during processing.
     */
    private fun analyzeImage() {
        val uri = selectedImageUri ?: return

        showLoading(true)

        lifecycleScope.launch {
            try {
                // Decode the image from its URI to a Bitmap (runs on IO thread)
                val bitmap = withContext(Dispatchers.IO) {
                    val inputStream = contentResolver.openInputStream(uri)
                    BitmapFactory.decodeStream(inputStream)
                }

                if (bitmap == null) {
                    withContext(Dispatchers.Main) {
                        showLoading(false)
                        Toast.makeText(
                            this@MainActivity,
                            "Failed to load image",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                    return@launch
                }

                // Run on-device TFLite inference (on IO thread for performance)
                val result = withContext(Dispatchers.IO) {
                    classifier.classify(bitmap)
                }

                withContext(Dispatchers.Main) {
                    showLoading(false)

                    if (result.success) {
                        // Save the scan result to history (if enabled in settings)
                        if (AppSettings.isSaveHistoryEnabled(this@MainActivity)) {
                            ScanHistoryManager.addScan(
                                this@MainActivity,
                                ScanRecord(
                                    predictedClass = result.predictedClass,
                                    confidence = result.confidence,
                                    probGood = result.probGood,
                                    probDefective = result.probDefective,
                                    imageUri = uri.toString()
                                )
                            )
                        }

                        // Navigate to the results screen with all prediction data
                        val intent = Intent(this@MainActivity, ResultActivity::class.java).apply {
                            putExtra(ResultActivity.EXTRA_IMAGE_URI, uri.toString())
                            putExtra(ResultActivity.EXTRA_PREDICTED_CLASS, result.predictedClass)
                            putExtra(ResultActivity.EXTRA_CONFIDENCE, result.confidence)
                            putExtra(ResultActivity.EXTRA_PROB_GOOD, result.probGood)
                            putExtra(ResultActivity.EXTRA_PROB_DEFECTIVE, result.probDefective)
                        }
                        startActivity(intent)
                        overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
                    } else {
                        // Show error message from the classifier
                        Toast.makeText(
                            this@MainActivity,
                            result.error ?: "Classification failed",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
                withContext(Dispatchers.Main) {
                    showLoading(false)
                    Toast.makeText(
                        this@MainActivity,
                        "Error: ${e.message}",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        }
    }

    // ─────────────────────────────────────────────
    // UI HELPERS
    // ─────────────────────────────────────────────

    /**
     * Show or hide the loading overlay and disable buttons during analysis.
     *
     * @param show True to show the overlay and disable interaction.
     */
    private fun showLoading(show: Boolean) {
        binding.loadingOverlay.isVisible = show
        binding.btnAnalyze.isEnabled = !show
        binding.btnCamera.isEnabled = !show
        binding.btnGallery.isEnabled = !show
    }
}
