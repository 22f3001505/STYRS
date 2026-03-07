package com.styrs.solarcell.ui

import android.animation.ObjectAnimator
import android.net.Uri
import android.os.Bundle
import android.view.animation.AccelerateDecelerateInterpolator
import android.view.animation.AlphaAnimation
import android.view.animation.AnimationSet
import android.view.animation.ScaleAnimation
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import coil.load
import com.styrs.solarcell.R
import com.styrs.solarcell.databinding.ActivityResultBinding

/**
 * ResultActivity — Displays the AI Classification Result
 * ========================================================
 *
 * Version : 3.0
 *
 * This activity is launched after the TFLite model has classified a solar
 * cell image. It receives the prediction data (class, confidence, probabilities)
 * via Intent extras and presents them with polished animations:
 *
 *   - The result card scales up and fades in (400ms)
 *   - Probability bars animate from 0% to their target values (800ms)
 *   - A colour-coded verdict (green for Good, red for Defective)
 *   - A context-specific recommendation for the quality inspector
 *
 * Navigation:
 *   MainActivity → [this] ResultActivity
 *                   ↓  (user taps "NEW SCAN")
 *                  finish() → back to MainActivity
 */
class ResultActivity : AppCompatActivity() {

    /** View binding — provides type-safe access to all UI elements */
    private lateinit var binding: ActivityResultBinding

    companion object {
        /** Intent extra keys — used by MainActivity to pass prediction data */
        const val EXTRA_IMAGE_URI = "extra_image_uri"
        const val EXTRA_PREDICTED_CLASS = "extra_predicted_class"
        const val EXTRA_CONFIDENCE = "extra_confidence"
        const val EXTRA_PROB_GOOD = "extra_prob_good"
        const val EXTRA_PROB_DEFECTIVE = "extra_prob_defective"
    }

    // ─────────────────────────────────────────────
    // LIFECYCLE
    // ─────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityResultBinding.inflate(layoutInflater)
        setContentView(binding.root)

        displayResults()      // Populate the UI with prediction data
        setupClickListeners() // Wire up the "NEW SCAN" button
        animateResults()      // Play entrance animations
    }

    // ─────────────────────────────────────────────
    // DISPLAY RESULTS
    // ─────────────────────────────────────────────

    /**
     * Extract prediction data from the Intent and populate all UI elements.
     *
     * The result card colour, icon, label, and recommendation text change
     * depending on whether the prediction is "Good" or "Defective".
     */
    private fun displayResults() {
        // Extract data that was passed from MainActivity
        val imageUri = intent.getStringExtra(EXTRA_IMAGE_URI)
        val predictedClass = intent.getStringExtra(EXTRA_PREDICTED_CLASS) ?: "Unknown"
        val confidence = intent.getDoubleExtra(EXTRA_CONFIDENCE, 0.0)
        val probGood = intent.getDoubleExtra(EXTRA_PROB_GOOD, 0.0)
        val probDefective = intent.getDoubleExtra(EXTRA_PROB_DEFECTIVE, 0.0)

        // Display the analysed image using Coil (with a smooth crossfade)
        imageUri?.let {
            binding.imageResult.load(Uri.parse(it)) {
                crossfade(true)
            }
        }

        // Determine the prediction outcome and style the result card accordingly
        val isGood = predictedClass.equals("Good", ignoreCase = true)

        if (isGood) {
            // ✅ Good result — green styling with positive recommendation
            binding.resultCard.background = ContextCompat.getDrawable(this, R.drawable.bg_result_good)
            binding.txtResultIcon.text = "✅"
            binding.txtResultLabel.text = "HEALTHY CELL"
            binding.txtResultLabel.setTextColor(ContextCompat.getColor(this, R.color.status_healthy))
            binding.txtRecommendation.text = getString(R.string.recommendation_good)
            binding.txtRecommendation.setTextColor(ContextCompat.getColor(this, R.color.status_healthy))
        } else {
            // ⚠️ Defective result — red styling with warning recommendation
            binding.resultCard.background = ContextCompat.getDrawable(this, R.drawable.bg_result_defective)
            binding.txtResultIcon.text = "⚠️"
            binding.txtResultLabel.text = "DEFECTIVE"
            binding.txtResultLabel.setTextColor(ContextCompat.getColor(this, R.color.status_critical))
            binding.txtRecommendation.text = getString(R.string.recommendation_defective)
            binding.txtRecommendation.setTextColor(ContextCompat.getColor(this, R.color.status_warning))
        }

        // Display the overall confidence as a formatted percentage
        binding.txtConfidence.text = String.format("Confidence: %.1f%%", confidence * 100)

        // Calculate integer percentages for the probability progress bars
        val goodPercent = (probGood * 100).toInt()
        val defectivePercent = (probDefective * 100).toInt()

        binding.txtGoodPercent.text = String.format("%d%%", goodPercent)
        binding.txtDefectivePercent.text = String.format("%d%%", defectivePercent)

        // Set initial progress to 0 — the actual values will be animated in later
        binding.progressGood.progress = 0
        binding.progressDefective.progress = 0

        // Store the target values using the View's tag (retrieved during animation)
        binding.progressGood.tag = goodPercent
        binding.progressDefective.tag = defectivePercent
    }

    // ─────────────────────────────────────────────
    // ANIMATIONS
    // ─────────────────────────────────────────────

    /**
     * Play entrance animations for the result card and probability bars.
     *
     * Timeline:
     *   0ms   — Result card scales up (0.8x→1.0x) and fades in (400ms)
     *   400ms — Probability bars animate from 0% to their target values (800ms)
     *
     * The staggered timing creates a polished, sequential reveal effect
     * that draws the user's eye to the most important information first
     * (the verdict) and then to the supporting details (probabilities).
     */
    private fun animateResults() {
        // Card entrance: scale from 80% to 100% + fade from 0 to 1
        val scaleUp = ScaleAnimation(
            0.8f, 1f, 0.8f, 1f,
            ScaleAnimation.RELATIVE_TO_SELF, 0.5f,
            ScaleAnimation.RELATIVE_TO_SELF, 0.5f
        ).apply { duration = 400 }

        val fadeIn = AlphaAnimation(0f, 1f).apply { duration = 400 }

        val animSet = AnimationSet(true).apply {
            addAnimation(scaleUp)
            addAnimation(fadeIn)
            interpolator = AccelerateDecelerateInterpolator()
        }
        binding.resultCard.startAnimation(animSet)

        // After the card finishes its entrance, animate the probability bars
        binding.resultCard.postDelayed({
            val goodTarget = binding.progressGood.tag as? Int ?: 0
            val defectiveTarget = binding.progressDefective.tag as? Int ?: 0

            // Animate the "Good" progress bar from 0 to its target percentage
            ObjectAnimator.ofInt(binding.progressGood, "progress", 0, goodTarget).apply {
                duration = 800
                interpolator = AccelerateDecelerateInterpolator()
                start()
            }

            // Animate the "Defective" progress bar from 0 to its target percentage
            ObjectAnimator.ofInt(binding.progressDefective, "progress", 0, defectiveTarget).apply {
                duration = 800
                interpolator = AccelerateDecelerateInterpolator()
                start()
            }
        }, 400)  // 400ms delay so bars animate after the card appears
    }

    // ─────────────────────────────────────────────
    // NAVIGATION
    // ─────────────────────────────────────────────

    /**
     * Set up the "NEW SCAN" button to return to MainActivity.
     */
    private fun setupClickListeners() {
        binding.btnNewScan.setOnClickListener {
            finish()  // Close this activity → returns to MainActivity
            overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
        }
    }
}
