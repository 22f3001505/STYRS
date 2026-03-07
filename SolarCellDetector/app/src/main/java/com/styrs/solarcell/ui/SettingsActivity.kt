package com.styrs.solarcell.ui

import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.styrs.solarcell.data.AppSettings
import com.styrs.solarcell.databinding.ActivitySettingsBinding

/**
 * SettingsActivity — App Configuration Screen
 * ==============================================
 *
 * Version : 3.0
 *
 * Allows users to configure app behaviour and view system information:
 *
 *   - Server URL field (disabled in v3.0 — app uses on-device inference)
 *   - Auto-Analyze toggle: automatically classify images after capture/selection
 *   - Save History toggle: persist scan records for the History screen
 *   - Connection status: shows "On-Device Mode" (no server needed)
 *
 * Settings are persisted using SharedPreferences (via AppSettings).
 *
 * Navigation:
 *   MainActivity → [this] SettingsActivity
 */
class SettingsActivity : AppCompatActivity() {

    /** View binding — provides type-safe access to all UI elements */
    private lateinit var binding: ActivitySettingsBinding

    // ─────────────────────────────────────────────
    // LIFECYCLE
    // ─────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySettingsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        loadSettings()     // Populate UI with current settings values
        setupListeners()   // Wire up toggles and buttons
    }

    // ─────────────────────────────────────────────
    // SETTINGS DISPLAY
    // ─────────────────────────────────────────────

    /**
     * Load current settings values and update the UI.
     *
     * Since v3.0, the app runs entirely on-device using TFLite, so the
     * server URL field is disabled and shows an informational message.
     * The connection test button is also disabled because no server
     * connection is needed.
     */
    private fun loadSettings() {
        // Load toggle states from SharedPreferences
        binding.switchAutoAnalyze.isChecked = AppSettings.isAutoAnalyzeEnabled(this)
        binding.switchSaveHistory.isChecked = AppSettings.isSaveHistoryEnabled(this)

        // Server URL field — disabled in on-device mode
        binding.editServerUrl.setText("On-Device (TFLite)")
        binding.editServerUrl.isEnabled = false       // Read-only in v3.0
        binding.btnSaveUrl.isEnabled = false           // No URL to save

        // Connection test button — replaced with on-device status
        binding.btnTestConnection.text = "ON-DEVICE MODE"
        binding.btnTestConnection.isEnabled = false    // No test needed

        // Show a green status message confirming on-device AI is active
        binding.txtConnectionStatus.text = "✅ Using on-device AI — no server needed"
        binding.txtConnectionStatus.setTextColor(
            getColor(com.styrs.solarcell.R.color.status_healthy)
        )
    }

    // ─────────────────────────────────────────────
    // EVENT LISTENERS
    // ─────────────────────────────────────────────

    /**
     * Wire up all interactive UI elements.
     */
    private fun setupListeners() {
        // Back button — return to MainActivity
        binding.btnBack.setOnClickListener { finish() }

        // Save URL button — informational toast in on-device mode
        binding.btnSaveUrl.setOnClickListener {
            Toast.makeText(this, "Using on-device model", Toast.LENGTH_SHORT).show()
        }

        // Auto-Analyze toggle — when enabled, images are automatically
        // classified immediately after being captured or selected
        binding.switchAutoAnalyze.setOnCheckedChangeListener { _, isChecked ->
            AppSettings.setAutoAnalyze(this, isChecked)
        }

        // Save History toggle — when enabled, scan results are saved to
        // SharedPreferences and displayed in the History screen
        binding.switchSaveHistory.setOnCheckedChangeListener { _, isChecked ->
            AppSettings.setSaveHistory(this, isChecked)
        }
    }
}
