"""
Reusable Empty State Component and On-Demand Artifact Generators.

Handles missing simulation artifacts gracefully on Streamlit:
- Provides a short informative sentence of what the tab shows.
- If a prerequisite file is missing, explains which file is required.
- If the artifact can be generated, provides a 'Generate now' button with capped parameters:
    - Stress: 3 scenarios x 5 episodes x 100 cycles
    - Tournament: 5 episodes per pairwise cell
    - Validation: Resampled from timeseries.json
- Shows a spinner while executing, persists the artifact, and calls st.rerun().
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional
import streamlit as st


def generate_stress_artifact(run_dir: str) -> None:
    """
    Generates stress_report.json with capped parameters:
    3 scenarios x 5 episodes x 100 cycles.
    """
    from src.scenarios.stress import StressScenarioRunner

    run_id = os.path.basename(os.path.normpath(run_dir))
    base_dir = os.path.dirname(os.path.normpath(run_dir))

    seed = 42
    summary_path = os.path.join(run_dir, "run_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                s_data = json.load(f)
            seed = int(s_data.get("seed", 42))
        except Exception:
            seed = 42

    runner = StressScenarioRunner(
        episodes=5,
        cycles_per_episode=100,
        seed=seed,
        output_dir=base_dir,
    )
    runner.run_suite(
        scenarios=["flash_crash", "informed_flow", "liquidity_withdrawal"],
        mm_types=["MarketMakerAgent", "AvellanedaStoikovMM"],
        run_id=run_id,
    )

    # Merge into run_summary.json without clobbering
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
            summary["stress_test"] = {
                "scenarios": ["flash_crash", "informed_flow", "liquidity_withdrawal"],
                "mm_types": ["MarketMakerAgent", "AvellanedaStoikovMM"],
                "episodes": 5,
                "cycles": 100,
                "seed": seed,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
        except Exception:
            pass


def generate_tournament_artifact(run_dir: str) -> None:
    """
    Generates equilibrium_analysis.json with capped parameters:
    5 episodes per pairwise cell (4 strategies x 50 cycles).
    """
    from scripts.run_tournament import run_tournament

    run_id = os.path.basename(os.path.normpath(run_dir))
    base_dir = os.path.dirname(os.path.normpath(run_dir))

    seed = 42
    summary_path = os.path.join(run_dir, "run_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                s_data = json.load(f)
            seed = int(s_data.get("seed", 42))
        except Exception:
            seed = 42

    run_tournament(
        episodes=5,
        cycles_per_episode=50,
        seed=seed,
        run_id=run_id,
        output_dir=base_dir,
        verbose=False,
    )


def generate_validation_artifact(run_dir: str) -> None:
    """
    Generates validation_report.json computed directly from timeseries.json.
    """
    from src.validation.stylized_facts import validate_stylized_facts

    run_id = os.path.basename(os.path.normpath(run_dir))
    base_dir = os.path.dirname(os.path.normpath(run_dir))

    ts_path = os.path.join(run_dir, "timeseries.json")
    if not os.path.exists(ts_path):
        raise FileNotFoundError(f"Prerequisite file 'timeseries.json' is missing in {run_dir}")

    validate_stylized_facts(
        run_id=run_id,
        data_dir=base_dir,
        resample_interval=5,
        save_report=True,
    )


def render_empty_state(
    tab_name: str,
    description: str,
    artifact_name: str,
    run_dir: Optional[str] = None,
    prerequisite_filename: Optional[str] = None,
    generator_fn: Optional[Callable[[str], Any]] = None,
    button_label: Optional[str] = None,
    spinner_text: Optional[str] = None,
) -> None:
    """
    Renders a unified empty state card for dashboard tabs lacking expected artifacts.

    Parameters
    ----------
    tab_name : str
        Display title of the tab view.
    description : str
        Short informative sentence explaining what the tab view displays.
    artifact_name : str
        Name of the missing artifact JSON file (e.g. 'stress_report.json').
    run_dir : str | None
        Directory of the selected simulation run.
    prerequisite_filename : str | None
        Name of a prerequisite file that must exist before artifact generation.
    generator_fn : Callable[[str], Any] | None
        Function to invoke to generate the artifact if prerequisites are satisfied.
    button_label : str | None
        Custom label for the generation button.
    spinner_text : str | None
        Custom text displayed during generation execution.
    """
    if not run_dir:
        st.info("ℹ️ No simulation run directory selected. Please select a past run from the sidebar or launch a new run.")
        return

    run_id = os.path.basename(os.path.normpath(run_dir))

    with st.container(border=True):
        st.markdown(f"**{tab_name}:** {description}")

        # Check prerequisite file
        if prerequisite_filename is not None:
            prereq_path = os.path.join(run_dir, prerequisite_filename)
            if not os.path.exists(prereq_path):
                st.info(
                    f"ℹ️ **Prerequisite File Missing:** This tab requires `{prerequisite_filename}`, "
                    f"which was not logged for run `{run_id}`. "
                    f"It must be recorded during simulation execution."
                )
                return

        # Prerequisite satisfied (or none required)
        if generator_fn is not None:
            st.caption(f"Artifact `{artifact_name}` is not yet compiled for this run. You can generate it now with optimized parameters.")
            label = button_label or f"⚡ Generate {artifact_name} Now"
            btn_key = f"btn_generate_{artifact_name}_{run_id}"

            if st.button(label, key=btn_key, type="primary"):
                spin = spinner_text or f"Generating {artifact_name} for run '{run_id}'..."
                with st.spinner(spin):
                    try:
                        generator_fn(run_dir)
                        st.success(f"Successfully compiled `{artifact_name}`!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to generate `{artifact_name}`: {e}")
        else:
            st.info(
                f"ℹ️ **Artifact Missing:** `{artifact_name}` is not available for run `{run_id}`. "
                f"It cannot be synthesized post-hoc and must be captured during simulation execution."
            )
