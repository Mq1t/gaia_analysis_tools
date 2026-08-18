"""Tests for epoch_photometry.py

Tests cover the phase calculation function, and lightcurve/lomb_scargle/pdm's input validation, release-dependent 
column requirements, error-bar and reject-flag handling, and saving behavior. Matplotlib rendering is patched out 
so tests run without a display.
"""

import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from gaia_analysis_tools.epoch_photometry import phase, lightcurve, lomb_scargle, pdm

# Fixtures
@pytest.fixture
def epoch_df():
    """Returns a minimal DR3-style epoch photometry DataFrame for lightcurve tests.

    Returns:
        pd.DataFrame: DataFrame with required lightcurve columns (no flux/flux_error columns).
    """
    n = 10
    return pd.DataFrame({
        "g_transit_time": np.linspace(0, 100, n),
        "g_transit_mag": np.random.uniform(12, 13, n),
        "bp_obs_time": np.linspace(0, 100, n),
        "bp_mag": np.random.uniform(12.5, 13.5, n),
        "rp_obs_time": np.linspace(0, 100, n),
        "rp_mag": np.random.uniform(11.5, 12.5, n),
        "variability_flag_g_reject": [False] * n,
        "variability_flag_bp_reject": [False] * n,
        "variability_flag_rp_reject": [False] * n,
    })

@pytest.fixture
def epoch_df_with_errors(epoch_df):
    """Returns epoch_df plus the flux/flux_error columns required when error=True.

    Args:
        epoch_df (pd.DataFrame): Base DR3-style photometry DataFrame fixture.

    Returns:
        pd.DataFrame: epoch_df with g/bp/rp flux and flux_error columns added.
    """
    n = len(epoch_df)
    df = epoch_df.copy()
    df["g_transit_flux"] = np.random.uniform(100, 200, n)
    df["g_transit_flux_error"] = np.random.uniform(1, 5, n)
    df["bp_flux"] = np.random.uniform(100, 200, n)
    df["bp_flux_error"] = np.random.uniform(1, 5, n)
    df["rp_flux"] = np.random.uniform(100, 200, n)
    df["rp_flux_error"] = np.random.uniform(1, 5, n)
    return df

@pytest.fixture
def epoch_df_dr4():
    """Returns a minimal DR4-style epoch photometry DataFrame (different G-band column names).

    Returns:
        pd.DataFrame: DataFrame with 'g_mag'/'g_obs_time' instead of the DR3 G-band columns.
    """
    n = 10
    return pd.DataFrame({
        "g_mag": np.random.uniform(12, 13, n),
        "g_obs_time": np.linspace(0, 100, n),
        "bp_obs_time": np.linspace(0, 100, n),
        "bp_mag": np.random.uniform(12.5, 13.5, n),
        "rp_obs_time": np.linspace(0, 100, n),
        "rp_mag": np.random.uniform(11.5, 12.5, n),
    })

@pytest.fixture
def time_series():
    """Returns a time and magnitude array for lomb_scargle and pdm tests.

    Returns:
        tuple: (t, mag) as pandas Series with 50 evenly spaced points.
    """
    t = pd.Series(np.linspace(0, 50, 50))
    mag = pd.Series(np.sin(2 * np.pi * t / 5.0))
    return t, mag


# phase
@pytest.mark.parametrize("t, T_0, P, expected", [
    (np.array([5.0]), 5.0, 3.0, 0.0),  # at reference epoch
    (np.array([8.0]), 5.0, 3.0, 0.0),  # one full period wraps back to zero
    (np.array([6.5]), 5.0, 3.0, 0.5),  # half a period
])
def test_phase_values(t, T_0, P, expected):
    """Checks phase at the reference epoch, after a full period, and at half a period."""
    result = phase(t, T_0, P)
    assert result[0] == pytest.approx(expected)

def test_phase_output_in_range_and_is_array():
    """Checks that phase values stay in [0, 1) across multiple periods, and returns a numpy array."""
    t = np.array([0.0, 1.0, 2.5, 10.0, 100.0])
    result = phase(t, T_0=0.0, P=3.0)
    assert isinstance(result, np.ndarray)
    assert np.all(result >= 0.0) and np.all(result < 1.0)


# lightcurve: input validation
def test_lightcurve_raises_type_error_for_non_dataframe():
    """Checks that lightcurve raises TypeError when given a non-DataFrame input."""
    with pytest.raises(TypeError):
        lightcurve([1, 2, 3])

def test_lightcurve_raises_key_error_for_missing_columns():
    """Checks that lightcurve raises KeyError when required columns are missing."""
    df = pd.DataFrame({"ra": [1, 2], "dec": [3, 4]})
    with pytest.raises(KeyError):
        lightcurve(df)

# lightcurve: plotting modes
def test_lightcurve_runs_overplot_mode(epoch_df):
    """Checks that lightcurve runs without error in overplot mode."""
    with patch("matplotlib.pyplot.show"):
        lightcurve(epoch_df, overplot=True)

def test_lightcurve_runs_subplot_mode(epoch_df):
    """Checks that lightcurve runs without error in subplot mode."""
    with patch("matplotlib.pyplot.show"):
        lightcurve(epoch_df, overplot=False)

def test_lightcurve_runs_with_period(epoch_df):
    """Checks that lightcurve runs without error when period is provided (phase folding)."""
    with patch("matplotlib.pyplot.show"):
        lightcurve(epoch_df, period=5.0)

def test_lightcurve_runs_with_rejectflags(epoch_df):
    """Checks that lightcurve runs without error when rejectflags is True (default DR3 release)."""
    with patch("matplotlib.pyplot.show"):
        lightcurve(epoch_df, rejectflags=True)

def test_lightcurve_runs_with_xlims_and_ylims(epoch_df):
    """Checks that lightcurve accepts xlims and ylims without error in overplot mode."""
    with patch("matplotlib.pyplot.show"):
        lightcurve(epoch_df, xlims=(0, 100), ylims=(11, 14))

def test_lightcurve_subplot_mode_respects_xlims_and_ylims(epoch_df):
    """Checks that xlims/ylims are applied per-axis without error when overplot=False."""
    with patch("matplotlib.pyplot.show"):
        lightcurve(epoch_df, overplot=False, xlims=(0, 100), ylims=(11, 14))

@pytest.mark.parametrize("disable_flag, cols_to_drop", [
    ("plot_g", ["g_transit_mag", "g_transit_time"]),
    ("plot_bp", ["bp_mag", "bp_obs_time"]),
    ("plot_rp", ["rp_mag", "rp_obs_time"]),
])
def test_lightcurve_disabling_a_band_does_not_require_its_columns(epoch_df, disable_flag, cols_to_drop):
    """Checks that plot_g/plot_bp/plot_rp=False, per the docstring, doesn't require that band's columns."""
    df = epoch_df.drop(columns=cols_to_drop)
    with patch("matplotlib.pyplot.show"):
        lightcurve(df, **{disable_flag: False})

# lightcurve: error bars
def test_lightcurve_runs_with_error_bars(epoch_df_with_errors):
    """Checks that lightcurve runs without error when error=True and flux/flux_error columns exist."""
    with patch("matplotlib.pyplot.show"):
        lightcurve(epoch_df_with_errors, error=True)

def test_lightcurve_error_true_missing_flux_columns_raises_key_error(epoch_df):
    """Checks that error=True raises KeyError when flux/flux_error columns aren't present."""
    with pytest.raises(KeyError):
        lightcurve(epoch_df, error=True)

# lightcurve: release-dependent columns
def test_lightcurve_dr4_uses_dr4_g_band_columns(epoch_df_dr4):
    """Checks that release='dr4' looks for 'g_mag'/'g_obs_time' instead of the DR3 G-band columns."""
    with patch("matplotlib.pyplot.show"):
        lightcurve(epoch_df_dr4, release="dr4")

def test_lightcurve_dr3_columns_missing_for_dr4_release_raises_key_error(epoch_df):
    """Checks that DR3-style G-band columns don't satisfy the DR4 column requirement."""
    with pytest.raises(KeyError):
        lightcurve(epoch_df, release="dr4")

def test_lightcurve_rejectflags_ignored_for_non_dr3_release(epoch_df_dr4):
    """Checks that rejectflags is only applied for release='dr3' and is silently ignored otherwise."""
    with patch("matplotlib.pyplot.show"):
        lightcurve(epoch_df_dr4, release="dr4", rejectflags=True)

# lightcurve: saving
def test_lightcurve_save_folder_none_skips_subfolder(epoch_df):
    """Checks that save_folder=None saves directly under file_name, with no subfolder created."""
    with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.savefig") as mock_savefig:
        lightcurve(epoch_df, save_plot=True, save_folder=None, file_name="my_plot")
    mock_savefig.assert_called_once()
    saved_path = mock_savefig.call_args[0][0]
    assert saved_path == "my_plot.pdf"

def test_lightcurve_save_folder_creates_subfolder(epoch_df):
    """Checks that save_plot writes into save_folder when one is given, creating it if needed."""
    with patch("matplotlib.pyplot.show"), patch("os.makedirs") as mock_makedirs, patch("matplotlib.pyplot.savefig") as mock_savefig:
        lightcurve(epoch_df, save_plot=True, save_folder="plots", file_name="my_plot")
    mock_makedirs.assert_called_once_with("plots", exist_ok=True)
    saved_path = mock_savefig.call_args[0][0]
    assert saved_path == os.path.join("plots", "my_plot.pdf")


# lomb_scargle
def test_lomb_scargle_returns_dataframe_with_expected_columns(time_series):
    """Checks that lomb_scargle returns a DataFrame with the periodogram columns."""
    t, mag = time_series
    result = lomb_scargle(t, mag)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["period", "power", "false alarm probability"]

def test_lomb_scargle_best_period_is_positive(time_series):
    """Checks that the best-fit period (highest power row) is a positive value."""
    t, mag = time_series
    result = lomb_scargle(t, mag)
    best_period = result.loc[result["power"].idxmax(), "period"]
    assert best_period > 0.0

def test_lomb_scargle_default_example_returns_dataframe():
    """Checks that lomb_scargle returns a periodogram DataFrame when called with no arguments."""
    result = lomb_scargle()
    assert isinstance(result, pd.DataFrame)
    assert "period" in result.columns

def test_lomb_scargle_period_column_respects_period_range(time_series):
    """Checks that the searched periods stay within a custom period_range."""
    t, mag = time_series
    result = lomb_scargle(t, mag, period_range=[1.0, 20.0])
    assert result["period"].min() >= 0.9
    assert result["period"].max() <= 20.1

def test_lomb_scargle_plot_runs_without_error(time_series):
    """Checks that lomb_scargle runs without error when plot is True."""
    t, mag = time_series
    with patch("matplotlib.pyplot.show"):
        lomb_scargle(t, mag, plot=True)

def test_lomb_scargle_save_data_writes_csv(time_series):
    """Checks that save_data=True saves the periodogram DataFrame to save_folder as a CSV."""
    t, mag = time_series
    with patch("os.makedirs"), patch("pandas.DataFrame.to_csv") as mock_to_csv:
        lomb_scargle(t, mag, save_data=True, data_file="my_ls", save_folder="results")
    saved_path = mock_to_csv.call_args[0][0]
    assert saved_path == os.path.join("results", "my_ls.csv")

def test_lomb_scargle_save_plot_writes_pdf(time_series):
    """Checks that plot=True with save_plot=True saves the periodogram plot to save_folder."""
    t, mag = time_series
    with patch("matplotlib.pyplot.show"), patch("os.makedirs"), patch("matplotlib.pyplot.savefig") as mock_savefig:
        lomb_scargle(t, mag, plot=True, save_plot=True, plot_file="my_ls_plot", save_folder="results")
    saved_path = mock_savefig.call_args[0][0]
    assert saved_path == os.path.join("results", "my_ls_plot.pdf")


# pdm
def test_pdm_returns_dataframe_with_expected_columns(time_series):
    """Checks that pdm returns a DataFrame with the documented period/frequency/theta columns."""
    t, mag = time_series
    result = pdm(t, mag)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["period", "frequency", "theta"]

def test_pdm_best_period_is_positive(time_series):
    """Checks that the best-fit period is a positive value."""
    t, mag = time_series
    result = pdm(t, mag)
    best_frequency = result.loc[result["theta"].idxmin(), "frequency"]
    best_period = 1.0 / best_frequency
    assert best_period > 0.0

def test_pdm_frequency_column_respects_freq_range(time_series):
    """Checks that pdm respects a custom freq_range when searching frequencies."""
    t, mag = time_series
    result = pdm(t, mag, freq_range=[0.1, 2.0, 0.01])
    assert result["frequency"].min() >= 0.1
    assert result["frequency"].max() <= 2.0

def test_pdm_plot_runs_without_error(time_series):
    """Checks that pdm runs without error when plot is True."""
    t, mag = time_series
    with patch("matplotlib.pyplot.show"):
        pdm(t, mag, plot=True)

def test_pdm_custom_bins_and_covers_returns_dataframe(time_series):
    """Checks that pdm accepts custom bins and covers and still returns a DataFrame."""
    t, mag = time_series
    result = pdm(t, mag, bins=30, covers=2)
    assert isinstance(result, pd.DataFrame)

def test_pdm_save_data_writes_csv(time_series):
    """Checks that save_data=True saves the PDM DataFrame to save_folder as a CSV."""
    t, mag = time_series
    with patch("os.makedirs"), patch("pandas.DataFrame.to_csv") as mock_to_csv:
        pdm(t, mag, save_data=True, data_file="my_pdm", save_folder="results")
    saved_path = mock_to_csv.call_args[0][0]
    assert saved_path == os.path.join("results", "my_pdm.csv")

def test_pdm_save_plot_writes_pdf(time_series):
    """Checks that plot=True with save_plot=True saves the PDM plot to save_folder."""
    t, mag = time_series
    with patch("matplotlib.pyplot.show"), patch("os.makedirs"), patch("matplotlib.pyplot.savefig") as mock_savefig:
        pdm(t, mag, plot=True, save_plot=True, plot_file="my_pdm_plot", save_folder="results")
    saved_path = mock_savefig.call_args[0][0]
    assert saved_path == os.path.join("results", "my_pdm_plot.pdf")