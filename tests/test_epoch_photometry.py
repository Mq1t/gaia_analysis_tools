"""Tests for epoch_photometry.py

Tests cover the phase calculation function and input validation for lightcurve, lomb_scargle, and pdm.
Matplotlib rendering is patched out so tests run without a display.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from gaiadr3_analysis.epoch_photometry import phase, lightcurve, lomb_scargle, pdm

# Fixtures
@pytest.fixture
def epoch_df():
    """Returns a minimal epoch photometry DataFrame for lightcurve tests.

    Returns:
        pd.DataFrame: DataFrame with required lightcurve columns.
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
def time_series():
    """Returns a time and magnitude array for lomb_scargle and pdm tests.

    Returns:
        tuple: (t, mag) as pandas Series with 50 evenly spaced points.
    """
    t = pd.Series(np.linspace(0, 50, 50))
    mag = pd.Series(np.sin(2 * np.pi * t / 5.0))
    return t, mag


# phase
def test_phase_output_in_range():
    """Checks that phase values are always in the interval [0, 1)."""
    t = np.array([0.0, 1.0, 2.5, 10.0, 100.0])
    result = phase(t, T_0=0.0, P=3.0)

    assert np.all(result >= 0.0)
    assert np.all(result < 1.0)

def test_phase_at_reference_epoch_is_zero():
    """Checks that phase is 0.0 when t equals T_0."""
    result = phase(np.array([5.0]), T_0=5.0, P=3.0)
    assert result[0] == pytest.approx(0.0)

def test_phase_one_full_period_is_zero():
    """Checks that t = T_0 + P wraps back to phase 0.0."""
    result = phase(np.array([8.0]), T_0=5.0, P=3.0)
    assert result[0] == pytest.approx(0.0)

def test_phase_half_period_is_point_five():
    """Checks that t = T_0 + P/2 gives phase 0.5."""
    result = phase(np.array([6.5]), T_0=5.0, P=3.0)
    assert result[0] == pytest.approx(0.5)

def test_phase_returns_numpy_array():
    """Checks that phase always returns a numpy array."""
    t = np.array([1.0, 2.0, 3.0])
    result = phase(t, T_0=0.0, P=2.0)
    assert isinstance(result, np.ndarray)

def test_phase_handles_array_input():
    """Checks that phase correctly handles a multi-element array."""
    t = np.array([0.0, 1.0, 2.0, 3.0])
    result = phase(t, T_0=0.0, P=2.0)
    assert len(result) == 4


# lightcurve
def test_lightcurve_raises_type_error_for_non_dataframe():
    """Checks that lightcurve raises TypeError when given a non-DataFrame input."""
    with pytest.raises(TypeError):
        lightcurve([1, 2, 3])

def test_lightcurve_raises_key_error_for_missing_columns():
    """Checks that lightcurve raises KeyError when required columns are missing."""
    df = pd.DataFrame({"ra": [1, 2], "dec": [3, 4]})
    with pytest.raises(KeyError):
        lightcurve(df)

def test_lightcurve_runs_overplot_mode(epoch_df):
    """Checks that lightcurve runs without error in overplot mode.

    Args:
        epoch_df (pd.DataFrame): Sample photometry DataFrame fixture.
    """
    with patch("matplotlib.pyplot.show"):
        lightcurve(epoch_df, overplot=True)

def test_lightcurve_runs_subplot_mode(epoch_df):
    """Checks that lightcurve runs without error in subplot mode.

    Args:
        epoch_df (pd.DataFrame): Sample photometry DataFrame fixture.
    """
    with patch("matplotlib.pyplot.show"):
        lightcurve(epoch_df, overplot=False)

def test_lightcurve_runs_with_period(epoch_df):
    """Checks that lightcurve runs without error when period is provided.

    Args:
        epoch_df (pd.DataFrame): Sample photometry DataFrame fixture.
    """
    with patch("matplotlib.pyplot.show"):
        lightcurve(epoch_df, period=5.0)

def test_lightcurve_runs_with_rejectflags(epoch_df):
    """Checks that lightcurve runs without error when rejectflags is True.

    Args:
        epoch_df (pd.DataFrame): Sample photometry DataFrame fixture.
    """
    with patch("matplotlib.pyplot.show"):
        lightcurve(epoch_df, rejectflags=True)

def test_lightcurve_runs_with_xlims_and_ylims(epoch_df):
    """Checks that lightcurve accepts xlims and ylims without error.

    Args:
        epoch_df (pd.DataFrame): Sample photometry DataFrame fixture.
    """
    with patch("matplotlib.pyplot.show"):
        lightcurve(epoch_df, xlims=(0, 100), ylims=(11, 14))

@pytest.mark.parametrize("disable_flag, cols_to_drop", [
    ("plot_g", ["g_transit_mag", "g_transit_time"]),
    ("plot_bp", ["bp_mag", "bp_obs_time"]),
    ("plot_rp", ["rp_mag", "rp_obs_time"]),
])
def test_lightcurve_disabling_a_band_does_not_require_its_columns(epoch_df, disable_flag, cols_to_drop):
    """Checks that plot_g/plot_bp/plot_rp=False, per the docstring, doesn't require that band's columns.

    Args:
        epoch_df (pd.DataFrame): Sample photometry DataFrame fixture.
        disable_flag (str): Which of plot_g/plot_bp/plot_rp to set False.
        cols_to_drop (list[str]): That band's columns, removed from the DataFrame before calling.
    """
    df = epoch_df.drop(columns=cols_to_drop)
    with patch("matplotlib.pyplot.show"):
        lightcurve(df, **{disable_flag: False})

def test_lightcurve_save_folder_none_skips_subfolder(epoch_df):
    """Checks that save_folder=None saves directly under file_name, with no subfolder created."""
    with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.savefig") as mock_savefig:
        lightcurve(epoch_df, save_plot=True, save_folder=None, file_name="my_plot")
    mock_savefig.assert_called_once()
    saved_path = mock_savefig.call_args[0][0]
    assert saved_path == "my_plot.pdf"


# lomb_scargle
def test_lomb_scargle_returns_dataframe_with_expected_columns(time_series):
    """Checks that lomb_scargle returns a DataFrame with the periodogram columns.

    Args:
        time_series (tuple): Sample (t, mag) time series fixture.
    """
    t, mag = time_series
    result = lomb_scargle(t, mag)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["period", "power", "false alarm probability"]

def test_lomb_scargle_best_period_is_positive(time_series):
    """Checks that the best-fit period (highest power row) is a positive value.

    Args:
        time_series (tuple): Sample (t, mag) time series fixture.
    """
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
    """Checks that the searched periods stay within a custom period_range.

    Args:
        time_series (tuple): Sample (t, mag) time series fixture.
    """
    t, mag = time_series
    result = lomb_scargle(t, mag, period_range=[1.0, 20.0])

    assert result["period"].min() >= 0.9
    assert result["period"].max() <= 20.1

def test_lomb_scargle_plot_runs_without_error(time_series):
    """Checks that lomb_scargle runs without error when plot is True.

    Args:
        time_series (tuple): Sample (t, mag) time series fixture.
    """
    t, mag = time_series
    with patch("matplotlib.pyplot.show"):
        lomb_scargle(t, mag, plot=True)


# pdm
def test_pdm_returns_dataframe_with_expected_columns(time_series):
    """Checks that pdm returns a DataFrame with the documented period/frequency/theta columns.

    Args:
        time_series (tuple): Sample (t, mag) time series fixture.
    """
    t, mag = time_series
    result = pdm(t, mag)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["period", "frequency", "theta"]

def test_pdm_best_period_is_positive(time_series):
    """Checks that the best-fit period is a positive value.

    Args:
        time_series (tuple): Sample (t, mag) time series fixture.
    """
    t, mag = time_series
    result = pdm(t, mag)
    best_frequency = result.loc[result["theta"].idxmin(), "frequency"]
    best_period = 1.0 / best_frequency

    assert best_period > 0.0

def test_pdm_frequency_column_respects_freq_range(time_series):
    """Checks that pdm respects a custom freq_range when searching frequencies.

    Args:
        time_series (tuple): Sample (t, mag) time series fixture.
    """
    t, mag = time_series
    result = pdm(t, mag, freq_range=[0.1, 2.0, 0.01])

    assert result["frequency"].min() >= 0.1
    assert result["frequency"].max() <= 2.0

def test_pdm_plot_runs_without_error(time_series):
    """Checks that pdm runs without error when plot is True.

    Args:
        time_series (tuple): Sample (t, mag) time series fixture.
    """
    t, mag = time_series
    with patch("matplotlib.pyplot.show"):
        pdm(t, mag, plot=True)

def test_pdm_custom_bins_and_covers_returns_dataframe(time_series):
    """Checks that pdm accepts custom bins and covers and still returns a DataFrame.

    Args:
        time_series (tuple): Sample (t, mag) time series fixture.
    """
    t, mag = time_series
    result = pdm(t, mag, bins=30, covers=2)

    assert isinstance(result, pd.DataFrame)