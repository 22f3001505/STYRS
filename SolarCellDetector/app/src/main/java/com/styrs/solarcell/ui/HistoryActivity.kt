package com.styrs.solarcell.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.card.MaterialCardView
import com.styrs.solarcell.R
import com.styrs.solarcell.data.ScanHistoryManager
import com.styrs.solarcell.data.ScanRecord
import com.styrs.solarcell.databinding.ActivityHistoryBinding
import java.text.SimpleDateFormat
import java.util.*

/**
 * HistoryActivity — Scan History List
 * =====================================
 *
 * Version : 3.0
 *
 * Displays a scrollable list of all past solar cell scans, including:
 *   - Result icon (✅ for Good, ⚠️ for Defective)
 *   - Classification label
 *   - Confidence percentage
 *   - Timestamp of when the scan was performed
 *
 * Users can clear the entire history via a confirmation dialog.
 * History data is persisted using SharedPreferences (via ScanHistoryManager).
 *
 * Navigation:
 *   MainActivity → [this] HistoryActivity
 */
class HistoryActivity : AppCompatActivity() {

    /** View binding — provides type-safe access to all UI elements */
    private lateinit var binding: ActivityHistoryBinding

    // ─────────────────────────────────────────────
    // LIFECYCLE
    // ─────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityHistoryBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Back button returns to MainActivity
        binding.btnBack.setOnClickListener { finish() }

        // Clear history button (with confirmation dialog)
        binding.btnClearHistory.setOnClickListener { confirmClearHistory() }

        // Load and display the saved scan history
        loadHistory()
    }

    // ─────────────────────────────────────────────
    // DATA LOADING
    // ─────────────────────────────────────────────

    /**
     * Load scan history from SharedPreferences and display it.
     *
     * If history is empty, show an "empty state" message and hide the clear button.
     * Otherwise, populate the RecyclerView with scan records.
     */
    private fun loadHistory() {
        val history = ScanHistoryManager.getHistory(this)

        if (history.isEmpty()) {
            // Show empty state — no scans recorded yet
            binding.recyclerHistory.visibility = View.GONE
            binding.emptyState.visibility = View.VISIBLE
            binding.btnClearHistory.visibility = View.GONE
        } else {
            // Show the list of previous scans
            binding.recyclerHistory.visibility = View.VISIBLE
            binding.emptyState.visibility = View.GONE
            binding.btnClearHistory.visibility = View.VISIBLE
            binding.recyclerHistory.layoutManager = LinearLayoutManager(this)
            binding.recyclerHistory.adapter = HistoryAdapter(history)
        }
    }

    // ─────────────────────────────────────────────
    // CLEAR HISTORY
    // ─────────────────────────────────────────────

    /**
     * Show a confirmation dialog before clearing all scan history.
     *
     * This prevents accidental data loss — the user must explicitly
     * confirm that they want to delete all records.
     */
    private fun confirmClearHistory() {
        AlertDialog.Builder(this, R.style.AlertDialogTheme)
            .setTitle("Clear History")
            .setMessage("Are you sure you want to clear all scan history?")
            .setPositiveButton("Clear") { _, _ ->
                ScanHistoryManager.clearHistory(this)
                loadHistory()   // Refresh the UI to show empty state
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    // ─────────────────────────────────────────────
    // RECYCLERVIEW ADAPTER
    // ─────────────────────────────────────────────

    /**
     * Adapter for displaying scan history items in the RecyclerView.
     *
     * Each item shows the scan icon, classification label, confidence
     * percentage, and the date/time of the scan. Items have a staggered
     * fade-in animation for a polished appearance.
     */
    inner class HistoryAdapter(private val items: List<ScanRecord>) :
        RecyclerView.Adapter<HistoryAdapter.ViewHolder>() {

        inner class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
            val card: MaterialCardView = view.findViewById(R.id.historyCard)
            val icon: TextView = view.findViewById(R.id.txtIcon)
            val label: TextView = view.findViewById(R.id.txtLabel)
            val confidence: TextView = view.findViewById(R.id.txtConfidence)
            val timestamp: TextView = view.findViewById(R.id.txtTimestamp)
        }

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
            val view = LayoutInflater.from(parent.context)
                .inflate(R.layout.item_history, parent, false)
            return ViewHolder(view)
        }

        override fun onBindViewHolder(holder: ViewHolder, position: Int) {
            val scan = items[position]
            val isGood = scan.predictedClass.equals("Good", ignoreCase = true)

            // Set icon and label based on classification result
            holder.icon.text = if (isGood) "✅" else "⚠️"
            holder.label.text = if (isGood) "HEALTHY" else "DEFECTIVE"
            holder.label.setTextColor(
                ContextCompat.getColor(
                    holder.itemView.context,
                    if (isGood) R.color.status_healthy else R.color.status_critical
                )
            )

            // Display confidence as a percentage (e.g. "92.3%")
            holder.confidence.text = String.format("%.1f%%", scan.confidence * 100)

            // Format the timestamp to a human-readable date and time
            holder.timestamp.text = SimpleDateFormat(
                "MMM dd, yyyy  HH:mm", Locale.getDefault()
            ).format(Date(scan.timestamp))

            // Stagger animation — each card fades in with a slight delay
            // based on its position, creating a cascading effect
            holder.card.alpha = 0f                 // Start invisible
            holder.card.translationY = 30f         // Start slightly below
            holder.card.animate()
                .alpha(1f)                         // Fade in
                .translationY(0f)                  // Slide up to final position
                .setDuration(300)
                .setStartDelay((position * 50).toLong())  // 50ms delay per item
                .start()
        }

        override fun getItemCount() = items.size
    }
}
