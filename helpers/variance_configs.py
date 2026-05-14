from __future__ import annotations

from typing import Any

# ────────────────────────────────────────────────────────────────────────────
# Scoring configurations to explore
# Decay constants are intentionally absent from all configs — they are fixed
# at whatever value the scorer modules define.
# ────────────────────────────────────────────────────────────────────────────SCORING_CONFIGS: list[dict[str, Any]] = [

SCORING_CONFIGS: list[dict[str, Any]] = [
    # ── baseline ───────────────────────────────────────────────────────────
    {
        "name": "baseline",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 1, "ev_wait_time": 3, "missed_deadline": 2},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 3},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },

    # ══ EV metric weight sweeps ════════════════════════════════════════════

    # -- equal EV weights --------------------------------------------------
    {
        "name": "ev_equal_weights",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 1, "ev_wait_time": 1, "missed_deadline": 1},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 3},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },

    # -- missed deadlines dominate (moderate) ------------------------------
    {
        "name": "deadline_heavy",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 1, "ev_wait_time": 1, "missed_deadline": 10},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 3},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },

    # -- missed deadlines dominate (extreme) -------------------------------
    {
        "name": "deadline_extreme",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 1, "ev_wait_time": 1, "missed_deadline": 20},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 3},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },

    # -- missed deadlines dominate (nuclear) -------------------------------
    {
        "name": "deadline_nuclear",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 1, "ev_wait_time": 1, "missed_deadline": 50},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 3},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },

    # -- missed deadlines only (everything else zeroed out) ----------------
    {
        "name": "deadline_only",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 0, "delta_arrival": 0, "ev_wait_time": 0, "missed_deadline": 1},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 3},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },

    # -- missed deadlines only, EV group only ------------------------------
    {
        "name": "deadline_pure",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 0, "delta_arrival": 0, "ev_wait_time": 0, "missed_deadline": 1},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 3},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 0},
    },

    # -- wait time dominates (moderate) ------------------------------------
    {
        "name": "ev_wait_heavy",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 1, "ev_wait_time": 8, "missed_deadline": 2},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 3},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },

    # -- wait time dominates (extreme) -------------------------------------
    {
        "name": "ev_wait_extreme",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 1, "ev_wait_time": 15, "missed_deadline": 2},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 3},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },

    # -- wait time dominates (nuclear) -------------------------------------
    {
        "name": "ev_wait_nuclear",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 1, "ev_wait_time": 50, "missed_deadline": 2},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 3},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },

    # -- wait time only (EV group) -----------------------------------------
    {
        "name": "ev_wait_only",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 0, "delta_arrival": 0, "ev_wait_time": 1, "missed_deadline": 0},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 3},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },

    # -- path deviation heavy (moderate) -----------------------------------
    {
        "name": "detour_heavy",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 5, "delta_arrival": 1, "ev_wait_time": 3, "missed_deadline": 2},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 3},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },

    # -- path deviation heavy (extreme) ------------------------------------
    {
        "name": "detour_extreme",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 10, "delta_arrival": 1, "ev_wait_time": 3, "missed_deadline": 2},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 3},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },

    # -- path deviation heavy (nuclear) ------------------------------------
    {
        "name": "detour_nuclear",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 30, "delta_arrival": 1, "ev_wait_time": 3, "missed_deadline": 2},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 3},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },

    # -- path deviation only -----------------------------------------------
    {
        "name": "detour_only",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 0, "ev_wait_time": 0, "missed_deadline": 0},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 3},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },

    # -- path deviation pure (station zeroed) ------------------------------
    {
        "name": "detour_pure",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 0, "ev_wait_time": 0, "missed_deadline": 0},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 3},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 0},
    },

    # -- delta arrival heavy -----------------------------------------------
    {
        "name": "arrival_heavy",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 8, "ev_wait_time": 3, "missed_deadline": 2},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 3},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },

    # -- delta arrival extreme ---------------------------------------------
    {
        "name": "arrival_extreme",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 20, "ev_wait_time": 3, "missed_deadline": 2},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 3},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },

    # -- delta arrival only ------------------------------------------------
    {
        "name": "arrival_only",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 0, "delta_arrival": 1, "ev_wait_time": 0, "missed_deadline": 0},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 3},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },

    # -- no path deviation (ignore detours) --------------------------------
    {
        "name": "no_detour_penalty",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 0, "delta_arrival": 1, "ev_wait_time": 3, "missed_deadline": 2},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 3},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },

    # -- EV wait + deadline only (no routing metrics) ----------------------
    {
        "name": "ev_charging_only",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 0, "delta_arrival": 0, "ev_wait_time": 1, "missed_deadline": 1},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 3},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },

    # -- routing only (detour + arrival, no charging metrics) --------------
    {
        "name": "ev_routing_only",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 1, "ev_wait_time": 0, "missed_deadline": 0},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 3},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },

    # ══ Station metric weight sweeps ═══════════════════════════════════════

    # -- station equal weights ---------------------------------------------
    {
        "name": "station_equal_weights",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 1, "ev_wait_time": 3, "missed_deadline": 2},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 1},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },

    # -- utilization dominates (moderate) ----------------------------------
    {
        "name": "utilization_heavy",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 1, "ev_wait_time": 3, "missed_deadline": 2},
        "STATION_METRIC_WEIGHTS": {"utilization": 8, "expected_wait_time": 1},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },

    # -- utilization dominates (extreme) -----------------------------------
    {
        "name": "utilization_extreme",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 1, "ev_wait_time": 3, "missed_deadline": 2},
        "STATION_METRIC_WEIGHTS": {"utilization": 30, "expected_wait_time": 1},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },

    # -- utilization only --------------------------------------------------
    {
        "name": "utilization_only",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 1, "ev_wait_time": 3, "missed_deadline": 2},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 0},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },

    # -- utilization pure (station group only) -----------------------------
    {
        "name": "utilization_pure",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 1, "ev_wait_time": 3, "missed_deadline": 2},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 0},
        "GROUP_WEIGHTS":          {"ev": 0, "station": 1},
    },

    # -- station expected wait dominates (moderate) ------------------------
    {
        "name": "station_wait_heavy",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 1, "ev_wait_time": 3, "missed_deadline": 2},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 10},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },

    # -- station expected wait dominates (extreme) -------------------------
    {
        "name": "station_wait_extreme",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 1, "ev_wait_time": 3, "missed_deadline": 2},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 30},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },

    # -- station wait pure (station group only) ----------------------------
    {
        "name": "station_wait_pure",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 1, "ev_wait_time": 3, "missed_deadline": 2},
        "STATION_METRIC_WEIGHTS": {"utilization": 0, "expected_wait_time": 1},
        "GROUP_WEIGHTS":          {"ev": 0, "station": 1},
    },

    # ══ Group weight sweeps ════════════════════════════════════════════════

    # -- station-centric (moderate) ----------------------------------------
    {
        "name": "station_centric",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 1, "ev_wait_time": 3, "missed_deadline": 2},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 3},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 3},
    },

    # -- station-centric (extreme) -----------------------------------------
    {
        "name": "station_centric_extreme",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 1, "ev_wait_time": 3, "missed_deadline": 2},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 3},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 10},
    },

    # -- station-centric (nuclear) -----------------------------------------
    {
        "name": "station_centric_nuclear",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 1, "ev_wait_time": 3, "missed_deadline": 2},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 3},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 50},
    },

    # -- EV-centric (moderate) ---------------------------------------------
    {
        "name": "ev_centric",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 1, "ev_wait_time": 3, "missed_deadline": 2},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 3},
        "GROUP_WEIGHTS":          {"ev": 3, "station": 1},
    },

    # -- EV-centric (extreme) ----------------------------------------------
    {
        "name": "ev_centric_extreme",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 1, "ev_wait_time": 3, "missed_deadline": 2},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 3},
        "GROUP_WEIGHTS":          {"ev": 10, "station": 1},
    },

    # -- EV-centric (nuclear) ----------------------------------------------
    {
        "name": "ev_centric_nuclear",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 1, "ev_wait_time": 3, "missed_deadline": 2},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 3},
        "GROUP_WEIGHTS":          {"ev": 50, "station": 1},
    },

    # -- station only (EV group zeroed out) --------------------------------
    {
        "name": "station_only",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 1, "ev_wait_time": 3, "missed_deadline": 2},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 3},
        "GROUP_WEIGHTS":          {"ev": 0, "station": 1},
    },

    # -- EV only (station group zeroed out) --------------------------------
    {
        "name": "ev_only",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 1, "ev_wait_time": 3, "missed_deadline": 2},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 3},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 0},
    },

    # ══ Cross-dimension combinations ═══════════════════════════════════════

    # -- deadline + station focus ------------------------------------------
    {
        "name": "deadline_and_station_focus",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 1, "ev_wait_time": 1, "missed_deadline": 10},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 5},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 3},
    },

    # -- wait time (both EV and station) focus -----------------------------
    {
        "name": "all_wait_focus",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 1, "ev_wait_time": 8, "missed_deadline": 2},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 8},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },

    # -- wait time extreme (both EV and station) ---------------------------
    {
        "name": "all_wait_extreme",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 1, "ev_wait_time": 20, "missed_deadline": 2},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 20},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },

    # -- throughput focus: utilization + deadline, ignore detours ----------
    {
        "name": "throughput_focus",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 0, "delta_arrival": 1, "ev_wait_time": 2, "missed_deadline": 5},
        "STATION_METRIC_WEIGHTS": {"utilization": 5, "expected_wait_time": 1},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 2},
    },

    # -- throughput extreme: max utilization, deadline kills score ---------
    {
        "name": "throughput_extreme",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 0, "delta_arrival": 1, "ev_wait_time": 1, "missed_deadline": 20},
        "STATION_METRIC_WEIGHTS": {"utilization": 20, "expected_wait_time": 1},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 5},
    },

    # -- user comfort focus: low detour, low wait, no missed deadlines -----
    {
        "name": "user_comfort_focus",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 5, "delta_arrival": 3, "ev_wait_time": 8, "missed_deadline": 10},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 5},
        "GROUP_WEIGHTS":          {"ev": 3, "station": 1},
    },

    # -- user comfort extreme: punish every negative EV experience ---------
    {
        "name": "user_comfort_extreme",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 15, "delta_arrival": 10, "ev_wait_time": 20, "missed_deadline": 30},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 10},
        "GROUP_WEIGHTS":          {"ev": 5, "station": 1},
    },

    # -- infrastructure focus: utilization + station wait, ignore EV detour
    {
        "name": "infrastructure_focus",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 0, "delta_arrival": 1, "ev_wait_time": 2, "missed_deadline": 2},
        "STATION_METRIC_WEIGHTS": {"utilization": 5, "expected_wait_time": 5},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 5},
    },

    # -- infrastructure extreme: stations are everything -------------------
    {
        "name": "infrastructure_extreme",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 0, "delta_arrival": 0, "ev_wait_time": 1, "missed_deadline": 1},
        "STATION_METRIC_WEIGHTS": {"utilization": 10, "expected_wait_time": 10},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 20},
    },

    # -- overloaded stations: penalise queue heavily, reward spread --------
    {
        "name": "queue_averse",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 1, "ev_wait_time": 5, "missed_deadline": 3},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 15},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 3},
    },

    # -- flat across everything: no metric has any priority ----------------
    {
        "name": "fully_flat",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 1, "ev_wait_time": 1, "missed_deadline": 1},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 1},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },

    # -- routing vs charging: detour + utilization, nothing else ----------
    {
        "name": "routing_vs_capacity",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 1, "delta_arrival": 0, "ev_wait_time": 0, "missed_deadline": 0},
        "STATION_METRIC_WEIGHTS": {"utilization": 1, "expected_wait_time": 0},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },

    # -- worst case penalty: punish every metric at maximum ----------------
    {
        "name": "worst_case_penalty",
        "EV_METRIC_WEIGHTS":      {"path_deviation": 20, "delta_arrival": 20, "ev_wait_time": 20, "missed_deadline": 20},
        "STATION_METRIC_WEIGHTS": {"utilization": 20, "expected_wait_time": 20},
        "GROUP_WEIGHTS":          {"ev": 1, "station": 1},
    },
]
