package com.styrs.solarcell.data

import android.content.Context
import android.content.SharedPreferences
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

/**
 * ScanHistoryManager — Persistent Scan History Storage
 * =====================================================
 *
 * Version : 3.0
 *
 * This singleton manages the storage and retrieval of past solar cell
 * scan results.  History is persisted as a JSON string in SharedPreferences,
 * which survives app restarts but is cleared when the app is uninstalled.
 *
 * The history is capped at 50 records to prevent unbounded storage growth.
 * New scans are inserted at the front of the list (most recent first).
 *
 * Data format:
 *   List<ScanRecord> → serialised as JSON → stored in SharedPreferences
 *
 * Usage:
 *   ScanHistoryManager.addScan(context, scanRecord)     // Add a new scan
 *   val history = ScanHistoryManager.getHistory(context) // Retrieve all scans
 *   ScanHistoryManager.clearHistory(context)             // Delete all scans
 */
object ScanHistoryManager {

    // SharedPreferences configuration
    private const val PREFS_NAME = "scan_history"       // Preferences file name
    private const val KEY_HISTORY = "history_list"       // Key for the JSON string
    private const val MAX_HISTORY_SIZE = 50              // Maximum number of records to keep

    // Gson instance for JSON serialisation/deserialisation
    private val gson = Gson()

    /**
     * Get the SharedPreferences instance for scan history storage.
     */
    private fun getPrefs(context: Context): SharedPreferences {
        return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    }

    /**
     * Add a new scan record to the history.
     *
     * The new record is inserted at the front of the list (index 0),
     * so the most recent scan always appears first in the HistoryActivity.
     * If the history exceeds 50 records, the oldest one is removed.
     *
     * @param context Android context (needed for SharedPreferences access).
     * @param scan The scan record to add.
     */
    fun addScan(context: Context, scan: ScanRecord) {
        val history = getHistory(context).toMutableList()
        history.add(0, scan)   // Insert at the front (most recent first)

        // Trim to max size — remove the oldest entry if we exceed the limit
        if (history.size > MAX_HISTORY_SIZE) {
            history.removeAt(history.lastIndex)
        }

        // Serialise the list to JSON and save to SharedPreferences
        val json = gson.toJson(history)
        getPrefs(context).edit().putString(KEY_HISTORY, json).apply()
    }

    /**
     * Retrieve the full scan history, sorted most-recent-first.
     *
     * @param context Android context.
     * @return List of ScanRecords, or an empty list if no history exists.
     */
    fun getHistory(context: Context): List<ScanRecord> {
        val json = getPrefs(context).getString(KEY_HISTORY, null) ?: return emptyList()

        // Use TypeToken to preserve the generic type during deserialisation
        // (Gson needs this because of Java's type erasure with generics)
        val type = object : TypeToken<List<ScanRecord>>() {}.type
        return try {
            gson.fromJson(json, type)
        } catch (e: Exception) {
            // If the JSON is corrupted or has an incompatible format,
            // return an empty list rather than crashing the app
            emptyList()
        }
    }

    /**
     * Delete all scan history.
     *
     * Called from HistoryActivity when the user confirms the "Clear History" action.
     *
     * @param context Android context.
     */
    fun clearHistory(context: Context) {
        getPrefs(context).edit().remove(KEY_HISTORY).apply()
    }
}

/**
 * ScanRecord — Data class representing a single solar cell scan result.
 *
 * This is persisted as part of the scan history JSON array.
 *
 * @property timestamp      Unix timestamp (milliseconds) when the scan was performed.
 * @property predictedClass The AI classification result: "Good" or "Defective".
 * @property confidence     Model confidence level (0.0 to 1.0).
 * @property probGood       Raw probability that the cell is good.
 * @property probDefective  Raw probability that the cell is defective.
 * @property imageUri       Optional URI string of the analysed image (for display).
 */
data class ScanRecord(
    val timestamp: Long = System.currentTimeMillis(),
    val predictedClass: String,
    val confidence: Double,
    val probGood: Double,
    val probDefective: Double,
    val imageUri: String? = null
)
