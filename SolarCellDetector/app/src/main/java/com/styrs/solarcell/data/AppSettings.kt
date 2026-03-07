package com.styrs.solarcell.data

import android.content.Context
import android.content.SharedPreferences

/**
 * AppSettings — Centralised Settings Management
 * ================================================
 *
 * Version : 3.0
 *
 * This singleton object manages all user-configurable settings using
 * Android's SharedPreferences.  SharedPreferences stores key-value pairs
 * persistently on the device's filesystem (as an XML file).
 *
 * Settings managed:
 *   - Server URL   : The Flask API endpoint (legacy, unused in v3.0)
 *   - Auto-Analyze : Whether images are auto-classified after capture
 *   - Save History : Whether scan results are saved for the History screen
 *
 * Usage:
 *   val url = AppSettings.getServerUrl(context)
 *   AppSettings.setAutoAnalyze(context, true)
 *   val isEnabled = AppSettings.isAutoAnalyzeEnabled(context)
 */
object AppSettings {

    // SharedPreferences file name — stored at /data/data/com.styrs.solarcell/shared_prefs/
    private const val PREFS_NAME = "app_settings"

    // Preference keys — these are the keys used to store/retrieve values
    private const val KEY_SERVER_URL = "server_url"
    private const val KEY_AUTO_ANALYZE = "auto_analyze"
    private const val KEY_SAVE_HISTORY = "save_history"

    // Default values — used when the preference has never been set
    private const val DEFAULT_SERVER_URL = "http://192.168.0.6:5001/"

    /**
     * Get the SharedPreferences instance for this app.
     *
     * MODE_PRIVATE ensures that only this app can read/write these preferences.
     */
    private fun getPrefs(context: Context): SharedPreferences {
        return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    }

    // ─────────────────────────────────────────────
    // SERVER URL (Legacy — not used in v3.0 on-device mode)
    // ─────────────────────────────────────────────

    /**
     * Get the saved server URL.
     *
     * In v3.0, the app uses on-device TFLite inference, so this setting
     * is informational only.  Kept for potential future server fallback.
     */
    fun getServerUrl(context: Context): String {
        return getPrefs(context).getString(KEY_SERVER_URL, DEFAULT_SERVER_URL) ?: DEFAULT_SERVER_URL
    }

    /**
     * Save a new server URL.
     */
    fun setServerUrl(context: Context, url: String) {
        getPrefs(context).edit().putString(KEY_SERVER_URL, url).apply()
    }

    // ─────────────────────────────────────────────
    // AUTO-ANALYZE
    // ─────────────────────────────────────────────

    /**
     * Check if auto-analyze is enabled.
     *
     * When enabled, images are immediately sent to the classifier after
     * being captured by the camera or selected from the gallery, without
     * requiring the user to tap the "ANALYZE" button.
     *
     * Default: false (user must manually tap Analyze)
     */
    fun isAutoAnalyzeEnabled(context: Context): Boolean {
        return getPrefs(context).getBoolean(KEY_AUTO_ANALYZE, false)
    }

    /**
     * Enable or disable auto-analyze.
     */
    fun setAutoAnalyze(context: Context, enabled: Boolean) {
        getPrefs(context).edit().putBoolean(KEY_AUTO_ANALYZE, enabled).apply()
    }

    // ─────────────────────────────────────────────
    // SAVE HISTORY
    // ─────────────────────────────────────────────

    /**
     * Check if scan history saving is enabled.
     *
     * When enabled, each classification result is saved to SharedPreferences
     * and displayed in the HistoryActivity.  When disabled, scans are
     * ephemeral and lost when the app closes.
     *
     * Default: true (history is saved)
     */
    fun isSaveHistoryEnabled(context: Context): Boolean {
        return getPrefs(context).getBoolean(KEY_SAVE_HISTORY, true)
    }

    /**
     * Enable or disable scan history saving.
     */
    fun setSaveHistory(context: Context, enabled: Boolean) {
        getPrefs(context).edit().putBoolean(KEY_SAVE_HISTORY, enabled).apply()
    }
}
