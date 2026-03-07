package com.styrs.solarcell.ui

import android.annotation.SuppressLint
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.animation.AccelerateDecelerateInterpolator
import android.view.animation.AlphaAnimation
import android.view.animation.AnimationSet
import android.view.animation.ScaleAnimation
import androidx.appcompat.app.AppCompatActivity
import com.styrs.solarcell.databinding.ActivitySplashBinding

/**
 * SplashActivity — Animated App Launch Screen
 * ==============================================
 *
 * Version : 3.0
 *
 * This is the very first screen users see when the app launches.
 * It displays the STYRS logo with a polished sequence of animations:
 *
 *   1. Logo scales up from 50% to 100% and fades in (0ms – 800ms)
 *   2. Tagline fades in (600ms delay)
 *   3. Version number fades in (900ms delay)
 *   4. Auto-navigates to MainActivity after 2 seconds
 *
 * The @SuppressLint("CustomSplashScreen") annotation is used because
 * we're implementing our own custom splash screen instead of using
 * the Android 12+ SplashScreen API, which gives us more control
 * over the animation design.
 */
@SuppressLint("CustomSplashScreen")
class SplashActivity : AppCompatActivity() {

    /** View binding — provides type-safe access to all UI elements */
    private lateinit var binding: ActivitySplashBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySplashBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // ── Step 1: Animate the logo ──
        // Scale from 50% to 100% size while simultaneously fading in
        val scaleUp = ScaleAnimation(
            0.5f, 1f, 0.5f, 1f,                             // fromX, toX, fromY, toY
            ScaleAnimation.RELATIVE_TO_SELF, 0.5f,           // Pivot at centre X
            ScaleAnimation.RELATIVE_TO_SELF, 0.5f            // Pivot at centre Y
        ).apply { duration = 800 }

        val fadeIn = AlphaAnimation(0f, 1f).apply { duration = 800 }

        val animSet = AnimationSet(true).apply {
            addAnimation(scaleUp)
            addAnimation(fadeIn)
            interpolator = AccelerateDecelerateInterpolator()  // Smooth easing
        }

        binding.splashLogo.startAnimation(animSet)

        // ── Step 2: Fade in the tagline (after a short delay) ──
        Handler(Looper.getMainLooper()).postDelayed({
            val taglineFade = AlphaAnimation(0f, 1f).apply { duration = 500 }
            binding.splashTagline.alpha = 1f
            binding.splashTagline.startAnimation(taglineFade)
        }, 600)   // 600ms after logo starts animating

        // ── Step 3: Fade in the version number ──
        Handler(Looper.getMainLooper()).postDelayed({
            val versionFade = AlphaAnimation(0f, 1f).apply { duration = 400 }
            binding.splashVersion.alpha = 1f
            binding.splashVersion.startAnimation(versionFade)
        }, 900)   // 900ms after logo starts animating

        // ── Step 4: Navigate to the main screen ──
        Handler(Looper.getMainLooper()).postDelayed({
            startActivity(Intent(this, MainActivity::class.java))
            overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
            finish()   // Remove splash from the back stack
        }, 2000)  // Total splash duration: 2 seconds
    }
}
