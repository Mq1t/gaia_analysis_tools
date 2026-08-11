"""Tests for mean_photometry.py

Tests cover input validation for ra_vs_dec and pmra_vs_pmdec, unit tests for get_distance, get_magnitude, 
get_bprp, and gaussian, and smoke tests for plot_hr_diagram, hist, and fitted_hist. Matplotlib rendering 
is patched out so tests run without a display.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from gaiadr3_analysis.mean_photometry import (
    ra_vs_dec, pmra_vs_pmdec,
    get_distance, get_magnitude, get_bprp,
    plot_hr_diagram, hist, gaussian, fitted_hist,
)


# Fixtures
@pytest.fixture
def mean_df():
    """Returns a DataFrame with ra and dec columns for position plot tests.

    Returns:
        pd.DataFrame: DataFrame with ra and dec columns.
    """
    return pd.DataFrame({
        "ra": [10.0, 20.0, 30.0],
        "dec": [-5.0, 15.0, 25.0],
    })

@pytest.fixture
def proper_motion_df():
    """Returns a DataFrame with pmra and pmdec columns for proper motion tests.

    Returns:
        pd.DataFrame: DataFrame with pmra and pmdec columns.
    """
    return pd.DataFrame({
        "pmra": [0.1, 0.2, 0.3],
        "pmdec": [-0.1, 0.0, 0.1],
    })

@pytest.fixture
def hr_df():
    """Returns a DataFrame with the columns required by plot_hr_diagram.

    Returns:
        pd.DataFrame: DataFrame with parallax, phot_g_mean_mag,
            phot_bp_mean_mag, and phot_rp_mean_mag columns.
    """
    return pd.DataFrame({
        "parallax": [10.0, 20.0, 5.0],
        "phot_g_mean_mag": [8.0, 9.5, 11.0],
        "phot_bp_mean_mag": [8.5, 10.0, 11.5],
        "phot_rp_mean_mag": [7.5, 9.0, 10.5],
    })

@pytest.fixture
def distances():
    """Returns a pandas series of distance values for histogram tests.

    Returns:
        pd.Series: Distance values in parsecs.
    """
    return pd.Series([100.0, 200.0, 150.0, 300.0, 250.0])


# ra_vs_dec
def test_ra_vs_dec_raises_type_error_for_non_dataframe():
    """Checks that ra_vs_dec raises TypeError when given a non-DataFrame input."""
    with pytest.raises(TypeError):
        ra_vs_dec([1, 2, 3])

def test_ra_vs_dec_raises_key_error_for_missing_ra(mean_df):
    """Checks that ra_vs_dec raises KeyError when the ra column is missing.

    Args:
        mean_df (pd.DataFrame): Sample sky position DataFrame fixture.
    """
    with pytest.raises(KeyError):
        ra_vs_dec(mean_df.drop(columns=["ra"]))

def test_ra_vs_dec_raises_key_error_for_missing_dec(mean_df):
    """Checks that ra_vs_dec raises KeyError when the dec column is missing.

    Args:
        mean_df (pd.DataFrame): Sample sky position DataFrame fixture.
    """
    with pytest.raises(KeyError):
        ra_vs_dec(mean_df.drop(columns=["dec"]))

def test_ra_vs_dec_runs_without_error(mean_df):
    """Checks that ra_vs_dec completes without error on valid input.

    Args:
        mean_df (pd.DataFrame): Sample sky position DataFrame fixture.
    """
    with patch("matplotlib.pyplot.show"):
        ra_vs_dec(mean_df)

def test_ra_vs_dec_accepts_custom_title(mean_df):
    """Checks that ra_vs_dec accepts a custom title without raising an error.

    Args:
        mean_df (pd.DataFrame): Sample sky position DataFrame fixture.
    """
    with patch("matplotlib.pyplot.show"):
        ra_vs_dec(mean_df, title="My Custom Title")

def test_ra_vs_dec_accepts_xlim_and_ylim(mean_df):
    """Checks that ra_vs_dec accepts xlim and ylim without raising an error.

    Args:
        mean_df (pd.DataFrame): Sample sky position DataFrame fixture.
    """
    with patch("matplotlib.pyplot.show"):
        ra_vs_dec(mean_df, xlim=50.0, ylim=50.0)

def test_ra_vs_dec_save_plot_uses_given_save_folder(mean_df):
    """Checks that save_plot=True saves inside the given save_folder, per the docstring.

    Args:
        mean_df (pd.DataFrame): Sample sky position DataFrame fixture.
    """
    with patch("matplotlib.pyplot.show"), \
         patch("os.makedirs") as mock_makedirs, \
         patch("matplotlib.pyplot.savefig") as mock_savefig:
        ra_vs_dec(mean_df, save_plot=True, save_folder="myfolder")
    mock_makedirs.assert_called_once_with("myfolder", exist_ok=True)
    assert mock_savefig.call_args[0][0] == "myfolder/ra_vs_dec.pdf"


# pmra_vs_pmdec
def test_pmra_vs_pmdec_raises_type_error_for_non_dataframe():
    """Checks that pmra_vs_pmdec raises TypeError when given a non-DataFrame input."""
    with pytest.raises(TypeError):
        pmra_vs_pmdec("not a dataframe")

def test_pmra_vs_pmdec_raises_key_error_for_missing_pmra(proper_motion_df):
    """Checks that pmra_vs_pmdec raises KeyError when pmra column is missing.

    Args:
        proper_motion_df (pd.DataFrame): Sample proper motion DataFrame fixture.
    """
    with pytest.raises(KeyError):
        pmra_vs_pmdec(proper_motion_df.drop(columns=["pmra"]))

def test_pmra_vs_pmdec_raises_key_error_for_missing_pmdec(proper_motion_df):
    """Checks that pmra_vs_pmdec raises KeyError when pmdec column is missing.

    Args:
        proper_motion_df (pd.DataFrame): Sample proper motion DataFrame fixture.
    """
    with pytest.raises(KeyError):
        pmra_vs_pmdec(proper_motion_df.drop(columns=["pmdec"]))

def test_pmra_vs_pmdec_runs_without_error(proper_motion_df):
    """Checks that pmra_vs_pmdec completes without error on valid input.

    Args:
        proper_motion_df (pd.DataFrame): Sample proper motion DataFrame fixture.
    """
    with patch("matplotlib.pyplot.show"):
        pmra_vs_pmdec(proper_motion_df)

def test_pmra_vs_pmdec_accepts_custom_title(proper_motion_df):
    """Checks that pmra_vs_pmdec accepts a custom title without raising an error.

    Args:
        proper_motion_df (pd.DataFrame): Sample proper motion DataFrame fixture.
    """
    with patch("matplotlib.pyplot.show"):
        pmra_vs_pmdec(proper_motion_df, title="Proper Motion Plot")

def test_pmra_vs_pmdec_save_plot_uses_given_save_folder(proper_motion_df):
    """Checks that save_plot=True saves inside the given save_folder, per the docstring.

    Args:
        proper_motion_df (pd.DataFrame): Sample proper motion DataFrame fixture.
    """
    with patch("matplotlib.pyplot.show"), \
         patch("os.makedirs") as mock_makedirs, \
         patch("matplotlib.pyplot.savefig") as mock_savefig:
        pmra_vs_pmdec(proper_motion_df, save_plot=True, save_folder="myfolder")
    mock_makedirs.assert_called_once_with("myfolder", exist_ok=True)
    assert mock_savefig.call_args[0][0] == "myfolder/pmra_vs_pmdec.pdf"

def test_pmra_vs_pmdec_error_true_reports_all_missing_columns():
    """Checks that error=True reports pmra/pmdec as missing too, not just the _error columns,
    per the docstring ("KeyError: If the required columns are missing ('pmra', 'pmdec')")."""
    with pytest.raises(KeyError) as exc_info:
        pmra_vs_pmdec(pd.DataFrame(), error=True)
    reported = [col.strip() for col in str(exc_info.value).split(":")[-1].strip(" '\"").split(",")]
    for col in ("pmra", "pmdec", "pmra_error", "pmdec_error"):
        assert col in reported


# get_distance
def test_get_distance_known_value():
    """Checks that get_distance returns the correct distance for a known parallax."""
    result = get_distance(10.0)
    assert result == pytest.approx(100.0)

def test_get_distance_one_kiloparsec():
    """Checks that get_distance returns 1000 pc for a parallax of 1 mas."""
    result = get_distance(1.0)
    assert result == pytest.approx(1000.0)

def test_get_distance_returns_float():
    """Checks that get_distance returns a numeric result."""
    result = get_distance(5.0)
    assert isinstance(result, float)


# get_magnitude
def test_get_magnitude_at_10_parsecs():
    """Checks that apparent and absolute magnitude are equal at 10 pc."""
    result = get_magnitude(8.0, 10.0)
    assert result == pytest.approx(8.0)

def test_get_magnitude_known_value():
    """Checks that get_magnitude returns the correct absolute magnitude for known inputs."""
    result = get_magnitude(10.0, 100.0)

    assert result == pytest.approx(5.0)

def test_get_magnitude_farther_star_is_dimmer():
    """Checks that a more distant star has a lower (brighter) absolute magnitude."""
    m_near = get_magnitude(10.0, 50.0)
    m_far = get_magnitude(10.0, 500.0)
    assert m_near > m_far


# get_bprp
def test_get_bprp_known_value():
    """Checks that get_bprp returns the correct BP-RP colour index."""
    result = get_bprp(10.5, 9.0)
    assert result == pytest.approx(1.5)

def test_get_bprp_zero_for_equal_magnitudes():
    """Checks that get_bprp returns 0.0 when BP and RP magnitudes are equal."""
    result = get_bprp(9.0, 9.0)
    assert result == pytest.approx(0.0)

def test_get_bprp_negative_for_red_star():
    """Checks that get_bprp returns a negative value when RP is brighter than BP."""
    result = get_bprp(9.0, 10.0)
    assert result < 0.0


# gaussian
def test_gaussian_peak_at_mu():
    """Checks that gaussian returns its maximum at x = mu."""
    result_at_mu = gaussian(5.0, A=1.0, sigma=1.0, mu=5.0)
    result_offset = gaussian(6.0, A=1.0, sigma=1.0, mu=5.0)
    assert result_at_mu > result_offset

def test_gaussian_returns_positive():
    """Checks that gaussian returns a positive value for standard inputs."""
    result = gaussian(0.0, A=1.0, sigma=1.0, mu=0.0)
    assert result > 0.0

def test_gaussian_symmetry():
    """Checks that gaussian is symmetric around mu."""
    left = gaussian(4.0, A=1.0, sigma=1.0, mu=5.0)
    right = gaussian(6.0, A=1.0, sigma=1.0, mu=5.0)
    assert left == pytest.approx(right)


# plot_hr_diagram
def test_plot_hr_diagram_runs_without_error(hr_df):
    """Checks that plot_hr_diagram completes without error on the columns the docstring
    says are required ("at minimum parallax, phot_g_mean_mag, phot_bp_mean_mag, and phot_rp_mean_mag").

    Args:
        hr_df (pd.DataFrame): Sample HR diagram DataFrame fixture.
    """
    with patch("matplotlib.pyplot.show"):
        plot_hr_diagram(hr_df)

def test_plot_hr_diagram_drops_nan_rows():
    """Checks that plot_hr_diagram handles NaN values without raising an error."""
    df = pd.DataFrame({
        "parallax": [10.0, None, 5.0],
        "phot_g_mean_mag": [8.0, 9.5, None],
        "phot_bp_mean_mag": [8.5, 10.0, 11.5],
        "phot_rp_mean_mag": [7.5, 9.0, 10.5],
    })
    with patch("matplotlib.pyplot.show"):
        plot_hr_diagram(df)

def test_plot_hr_diagram_save_plot_uses_given_save_folder(hr_df):
    """Checks that save_plot=True saves inside the given save_folder, per the docstring.

    Args:
        hr_df (pd.DataFrame): Sample HR diagram DataFrame fixture.
    """
    full_df = hr_df.assign(
        parallax_error=0.1,
        phot_g_mean_flux=1.0, phot_g_mean_flux_error=0.1,
        phot_bp_mean_flux=1.0, phot_bp_mean_flux_error=0.1,
        phot_rp_mean_flux=1.0, phot_rp_mean_flux_error=0.1,
    )
    with patch("matplotlib.pyplot.show"), \
         patch("os.makedirs") as mock_makedirs, \
         patch("matplotlib.pyplot.savefig") as mock_savefig:
        plot_hr_diagram(full_df, save_plot=True, save_folder="myfolder")
    mock_makedirs.assert_called_once_with("myfolder", exist_ok=True)
    assert mock_savefig.call_args[0][0] == "myfolder/hr_diagram.pdf"


# hist
def test_hist_runs_without_error(distances):
    """Checks that hist completes without error on valid input.

    Args:
        distances (pd.Series): Sample distance values fixture.
    """
    with patch("matplotlib.pyplot.show"):
        hist(distances)

def test_hist_runs_with_parallax_conversion(distances):
    """Checks that hist runs without error when parallax conversion is enabled.

    Args:
        distances (pd.Series): Sample distance values fixture.
    """
    with patch("matplotlib.pyplot.show"):
        hist(distances, parallax=True)

def test_hist_accepts_custom_bin_count(distances):
    """Checks that hist accepts a custom bin count without raising an error.

    Args:
        distances (pd.Series): Sample distance values fixture.
    """
    with patch("matplotlib.pyplot.show"):
        hist(distances, bin_num=20)

def test_hist_save_plot_uses_given_save_folder(distances):
    """Checks that save_plot=True saves inside the given save_folder, per the docstring.

    Args:
        distances (pd.Series): Sample distance values fixture.
    """
    with patch("matplotlib.pyplot.show"), \
         patch("os.makedirs") as mock_makedirs, \
         patch("matplotlib.pyplot.savefig") as mock_savefig:
        hist(distances, save_plot=True, save_folder="myfolder")
    mock_makedirs.assert_called_once_with("myfolder", exist_ok=True)
    assert mock_savefig.call_args[0][0] == "myfolder/histogram.pdf"


# fitted_hist
def test_fitted_hist_runs_without_error(distances):
    """Checks that fitted_hist completes without error on valid input.

    Args:
        distances (pd.Series): Sample distance values fixture.
    """
    with patch("matplotlib.pyplot.show"):
        fitted_hist(distances, range=[50, 400])

def test_fitted_hist_runs_with_parallax_conversion():
    """Checks that fitted_hist runs without error when parallax conversion is enabled.

    Uses a larger synthetic parallax dataset centred around 5 mas so that curve_fit has a well-shaped histogram
    to converge on after 1000/parallax conversion.
    """
    rng = np.random.default_rng(42)
    parallax_values = pd.Series(rng.normal(loc=5.0, scale=0.5, size=200))
    with patch("matplotlib.pyplot.show"):
        fitted_hist(parallax_values, parallax=True, range=[100, 400])

def test_fitted_hist_save_plot_uses_given_save_folder(distances):
    """Checks that save_plot=True saves inside the given save_folder, per the docstring.

    Args:
        distances (pd.Series): Sample distance values fixture.
    """
    with patch("matplotlib.pyplot.show"), \
         patch("os.makedirs") as mock_makedirs, \
         patch("matplotlib.pyplot.savefig") as mock_savefig:
        fitted_hist(distances, range=[50, 400], save_plot=True, save_folder="myfolder")
    mock_makedirs.assert_called_once_with("myfolder", exist_ok=True)
    assert mock_savefig.call_args[0][0] == "myfolder/fitted_hist.pdf"