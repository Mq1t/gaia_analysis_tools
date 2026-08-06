"""Tests for gaia_input.py

Tests cover CSV loading, filtering logic, identifier resolution via SIMBAD, cluster lookup, ADQL/Datalink querying, 
coordinate-based star search, login handling, the interactive menu's routing logic, and the Gaia DR3/DR4 release
helpers.

All calls to SIMBAD and the Gaia archive (via astroquery) are mocked, so this runs fully offline and does not require 
network access or live credentials. Interactive input prompts are also mocked with scripted answers so the menu 
logic can be tested.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from gaiadr3_analysis.gaia_input import (
    gaia_designation,
    is_value_masked,
    query_simbad_by_gaia_ids,
    fetch_coordinates,
    gaia_login_prompt,
    resolve_id,
    find_cluster,
    query_by_adql,
    query_by_datalink,
    find_star,
    load_csv,
    apply_filter,
    get_dataframe,
)

# A 26-character filler used to build fake Datalink keys for tests
KEY_PREFIX = "A" * 26

class FakeTable(list):
    """A minimal stand-in for an astropy Table: a list of dict-like rows plus a colnames attribute.

    Used to mock SIMBAD query results (query_hierarchy, query_region, query_objects), which the
    code under test accesses via row["column"] and checks "column" in result.colnames.
    """
    def __init__(self, rows, colnames):
        super().__init__(rows)
        self.colnames = colnames


# Fixtures
@pytest.fixture
def sample_df():
    """Returns a minimal Gaia DataFrame for testing.

    Returns:
        pd.DataFrame: DataFrame with ra, dec, parallax, and pmra columns.
    """
    return pd.DataFrame({
        "ra": [10.0, 20.0, 30.0],
        "dec": [-5.0, 15.0, 25.0],
        "parallax": [0.5, 1.5, 2.5],
        "pmra": [0.1, 0.2, 0.3],
    })

@pytest.fixture
def sample_dict_of_df():
    """Returns a dict of DataFrames mimicking query_by_datalink output.

    Returns:
        dict[int, pd.DataFrame]: Mapping of fake Gaia IDs to epoch
        photometry DataFrames.
    """
    return {
        111: pd.DataFrame({"flux": [1.0, 5.0, 10.0]}),
        222: pd.DataFrame({"flux": [2.0, 6.0, 11.0]}),
    }

# gaia_designation
def test_gaia_designation_default_dr3():
    """Checks that gaia_designation defaults to a DR3-prefixed designation string."""
    assert gaia_designation(123456789) == "Gaia DR3 123456789"

def test_gaia_designation_dr4():
    """Checks that gaia_designation builds a DR4-prefixed designation string when requested."""
    assert gaia_designation(123456789, release="dr4") == "Gaia DR4 123456789"

# is_value_masked
def test_is_value_masked_true_for_masked_value():
    """Checks that a value with mask=True is reported as masked."""
    masked_value = MagicMock(mask=True)
    assert is_value_masked(masked_value) is True

def test_is_value_masked_false_for_unmasked_value():
    """Checks that a value with mask=False is reported as not masked."""
    unmasked_value = MagicMock(mask=False)
    assert is_value_masked(unmasked_value) is False

def test_is_value_masked_false_for_plain_value():
    """Checks that a plain value with no mask attribute at all is reported as not masked."""
    assert is_value_masked(10.5) is False

# query_simbad_by_gaia_ids
@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_query_simbad_by_gaia_ids_returns_result(mock_simbad):
    """Checks that query_simbad_by_gaia_ids returns SIMBAD's result table on success.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
    """
    mock_simbad.query_objects.return_value = FakeTable([{"main_id": "Star A"}], colnames=["main_id"])
    result = query_simbad_by_gaia_ids([111])
    assert result is not None
    assert result[0]["main_id"] == "Star A"

@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_query_simbad_by_gaia_ids_handles_exception(mock_simbad):
    """Checks that query_simbad_by_gaia_ids returns None and prints the given error_context on failure.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
    """
    mock_simbad.query_objects.side_effect = Exception("network error")
    with patch("builtins.print") as mock_print:
        result = query_simbad_by_gaia_ids([111], error_context="test lookup")
    assert result is None
    printed = " ".join(str(call.args) for call in mock_print.call_args_list)
    assert "test lookup" in printed

# fetch_coordinates
@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_fetch_coordinates_uses_simbad_position(mock_simbad):
    """Checks that fetch_coordinates uses SIMBAD's position when it has a usable one on file.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
    """
    mock_simbad.query_objects.return_value = FakeTable(
        [{"main_id": "Star A", "ra": 10.0, "dec": 20.0}], colnames=["main_id", "ra", "dec"],
    )
    result = fetch_coordinates([111])
    assert result == [("Star A", 10.0, 20.0)]

@patch("gaiadr3_analysis.gaia_input.Gaia")
@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_fetch_coordinates_falls_back_to_gaia_archive(mock_simbad, mock_gaia):
    """Checks that fetch_coordinates falls back to the Gaia archive when SIMBAD has no usable position.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
        mock_gaia: Mocked Gaia class from astroquery.
    """
    mock_simbad.query_objects.return_value = None
    mock_job = MagicMock()
    mock_job.get_results.return_value.to_pandas.return_value = pd.DataFrame({
        "source_id": [111], "ra": [10.0], "dec": [20.0],
    })
    mock_gaia.launch_job.return_value = mock_job

    with patch("builtins.print"):
        result = fetch_coordinates([111])

    assert result == [("111", 10.0, 20.0)]

@patch("gaiadr3_analysis.gaia_input.Gaia")
@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_fetch_coordinates_warns_when_not_found_anywhere(mock_simbad, mock_gaia):
    """Checks that an ID found in neither SIMBAD nor the Gaia archive is skipped with a warning.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
        mock_gaia: Mocked Gaia class from astroquery.
    """
    mock_simbad.query_objects.return_value = None
    mock_job = MagicMock()
    mock_job.get_results.return_value.to_pandas.return_value = pd.DataFrame(columns=["source_id", "ra", "dec"])
    mock_gaia.launch_job.return_value = mock_job

    with patch("builtins.print") as mock_print:
        result = fetch_coordinates([111])

    assert result == []
    printed = " ".join(str(call.args) for call in mock_print.call_args_list)
    assert "111" in printed

# load_csv
def test_load_csv_returns_dataframe(tmp_path):
    """Checks that load_csv returns a pandas DataFrame.

    Args:
        tmp_path: pytest built-in fixture providing a temporary directory.
    """
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("ra,dec\n10.0,-5.0\n20.0,15.0\n")
    result = load_csv(str(csv_file))
    assert isinstance(result, pd.DataFrame)

def test_load_csv_correct_columns(tmp_path):
    """Checks that load_csv preserves column names from the CSV file.

    Args:
        tmp_path: pytest built-in fixture providing a temporary directory.
    """
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("ra,dec,parallax\n10.0,-5.0,1.0\n")
    result = load_csv(str(csv_file))
    assert list(result.columns) == ["ra", "dec", "parallax"]

def test_load_csv_correct_row_count(tmp_path):
    """Checks that load_csv loads the correct number of rows.

    Args:
        tmp_path: pytest built-in fixture providing a temporary directory.
    """
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("ra,dec\n10.0,-5.0\n20.0,15.0\n30.0,25.0\n")
    result = load_csv(str(csv_file))
    assert len(result) == 3

def test_load_csv_file_not_found():
    """Checks that load_csv raises FileNotFoundError for a missing file."""
    with pytest.raises(FileNotFoundError):
        load_csv("nonexistent_file.csv")

# apply_filter (single DataFrame)
def test_apply_filter_valid_expression(sample_df):
    """Checks that apply_filter correctly filters rows using a valid expression.

    Args:
        sample_df (pd.DataFrame): Sample Gaia DataFrame fixture.
    """
    with patch("builtins.print"), patch("builtins.input", return_value="parallax > 1.0"):
        result = apply_filter(sample_df)
    assert len(result) == 2
    assert all(result["parallax"] > 1.0)

def test_apply_filter_empty_expression_returns_original(sample_df):
    """Checks that apply_filter returns the original DataFrame when skipped.

    Args:
        sample_df (pd.DataFrame): Sample Gaia DataFrame fixture.
    """
    with patch("builtins.print"), patch("builtins.input", return_value=""):
        result = apply_filter(sample_df)
    assert len(result) == len(sample_df)

def test_apply_filter_invalid_expression_returns_original(sample_df):
    """Checks that apply_filter returns the original DataFrame on invalid input.

    Args:
        sample_df (pd.DataFrame): Sample Gaia DataFrame fixture.
    """
    with patch("builtins.print"), patch("builtins.input", return_value="not_a_column > 1"):
        result = apply_filter(sample_df)
    assert len(result) == len(sample_df)

def test_apply_filter_exact_match(sample_df):
    """Checks that apply_filter returns only rows matching an exact condition.

    Args:
        sample_df (pd.DataFrame): Sample Gaia DataFrame fixture.
    """
    with patch("builtins.print"), patch("builtins.input", return_value="parallax == 0.5"):
        result = apply_filter(sample_df)
    assert len(result) == 1
    assert result["parallax"].iloc[0] == 0.5

# apply_filter (dict of DataFrames)
def test_apply_filter_dict_applies_to_every_entry(sample_dict_of_df):
    """Checks that apply_filter applies the same expression to every DataFrame in a dict.

    Args:
        sample_dict_of_df (dict[int, pd.DataFrame]): Sample dict-of-DataFrames fixture.
    """
    with patch("builtins.print"), patch("builtins.input", return_value="flux > 4.0"):
        result = apply_filter(sample_dict_of_df)
    assert set(result.keys()) == {111, 222}
    assert len(result[111]) == 2
    assert len(result[222]) == 2
    assert (result[111]["flux"] > 4.0).all()

def test_apply_filter_dict_empty_expression_returns_original(sample_dict_of_df):
    """Checks that apply_filter returns the original dict unchanged when the filter is skipped.

    Args:
        sample_dict_of_df (dict[int, pd.DataFrame]): Sample dict-of-DataFrames fixture.
    """
    with patch("builtins.print"), patch("builtins.input", return_value=""):
        result = apply_filter(sample_dict_of_df)
    assert result is sample_dict_of_df

def test_apply_filter_dict_invalid_expression_keeps_original_entry(sample_dict_of_df):
    """Checks that apply_filter keeps the original DataFrame for an entry when its expression is invalid.

    Args:
        sample_dict_of_df (dict[int, pd.DataFrame]): Sample dict-of-DataFrames fixture.
    """
    with patch("builtins.print"), patch("builtins.input", return_value="not_a_column > 1"):
        result = apply_filter(sample_dict_of_df)
    assert len(result[111]) == len(sample_dict_of_df[111])

def test_apply_filter_empty_dict_returns_empty_dict():
    """Checks that apply_filter returns an empty dict unchanged without prompting for input."""
    result = apply_filter({})
    assert result == {}

# resolve_id: numeric/direct-match branches
@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_resolve_id_numeric_string_returns_int(mock_simbad):
    """Checks that a purely numeric string identifier is returned as an int without a SIMBAD common name.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
    """
    mock_simbad.query_objectids.return_value = None
    with patch("builtins.print"):
        result = resolve_id("123456789", plot=False)
    assert result == 123456789

@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_resolve_id_int_input_returns_int(mock_simbad):
    """Checks that an integer identifier is returned as-is.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
    """
    mock_simbad.query_objectids.return_value = None
    with patch("builtins.print"):
        result = resolve_id(987654321, plot=False)
    assert result == 987654321

@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_resolve_id_numeric_id_prints_common_name(mock_simbad):
    """Checks that a numeric ID with a SIMBAD 'NAME ' entry prints that common name.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
    """
    mock_simbad.query_objectids.return_value = {"id": ["NAME Proxima Centauri", "Gaia DR3 123456789"]}
    with patch("builtins.print") as mock_print:
        result = resolve_id(123456789, plot=False)
    assert result == 123456789
    printed = " ".join(str(call.args) for call in mock_print.call_args_list)
    assert "Proxima Centauri" in printed

@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_resolve_id_direct_cross_match(mock_simbad):
    """Checks that a name with a direct Gaia DR3 cross-match resolves to a single int.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
    """
    mock_simbad.query_objectids.return_value = {"id": ["HD 209458", "Gaia DR3 1234567890123456789"]}
    with patch("builtins.print"):
        result = resolve_id("HD 209458", plot=False)
    assert result == 1234567890123456789

@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_resolve_id_no_match_at_all_returns_none(mock_simbad):
    """Checks that resolve_id returns None when neither a direct match nor children are found.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
    """
    mock_simbad.query_objectids.return_value = None
    mock_simbad.query_hierarchy.return_value = None
    with patch("builtins.print"):
        result = resolve_id("Nonexistent Object", plot=False)
    assert result is None

@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_resolve_id_empty_children_table_returns_none(mock_simbad):
    """Checks that resolve_id returns None when query_hierarchy returns an empty table.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
    """
    mock_simbad.query_objectids.return_value = None
    mock_simbad.query_hierarchy.return_value = FakeTable([], colnames=["main_id", "otype"])
    with patch("builtins.print"):
        result = resolve_id("Empty Object", plot=False)
    assert result is None

# resolve_id: release parameter (DR4)
@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_resolve_id_dr4_uses_dr4_designation_for_numeric_id(mock_simbad):
    """Checks that a numeric identifier is looked up with a Gaia DR4 designation when release='dr4'.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
    """
    mock_simbad.query_objectids.return_value = None
    with patch("builtins.print"):
        resolve_id(123456789, plot=False, release="dr4")

    called_name = mock_simbad.query_objectids.call_args[0][0]
    assert called_name == "Gaia DR4 123456789"

@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_resolve_id_dr4_unresolved_name_prints_note(mock_simbad):
    """Checks that an unresolved name under release='dr4' prints a note about missing DR4 cross-matches.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
    """
    mock_simbad.query_objectids.return_value = None
    mock_simbad.query_hierarchy.return_value = None
    with patch("builtins.print") as mock_print:
        result = resolve_id("Some Star", plot=False, release="dr4")

    assert result is None
    printed = " ".join(str(call.args) for call in mock_print.call_args_list)
    assert "DR4" in printed

# resolve_id: cluster children
@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_resolve_id_resolves_children_to_list(mock_simbad):
    """Checks that an identifier with no direct match but resolvable children returns a list of IDs.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
    """
    mock_simbad.query_objectids.return_value = None
    mock_simbad.query_hierarchy.return_value = FakeTable(
        [{"main_id": "Star A", "otype": "*"}, {"main_id": "Star B", "otype": "*"}],
        colnames=["main_id", "otype"],
    )
    mock_simbad.query_tap.return_value = [
        {"input_name": "Star A", "alias": "Gaia DR3 111"},
        {"input_name": "Star B", "alias": "Gaia DR3 222"},
    ]

    with patch("builtins.print"):
        result = resolve_id("NGC 188", plot=False)

    assert result == [111, 222]

@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_resolve_id_children_found_but_none_have_gaia_id(mock_simbad):
    """Checks that resolve_id returns an empty list when children exist but none have a Gaia DR3 cross-match.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
    """
    mock_simbad.query_objectids.return_value = None
    mock_simbad.query_hierarchy.return_value = FakeTable(
        [{"main_id": "Star A", "otype": "*"}, {"main_id": "Star B", "otype": "*"}],
        colnames=["main_id", "otype"],
    )
    mock_simbad.query_tap.return_value = []

    with patch("builtins.print"):
        result = resolve_id("Cluster", plot=False)

    assert result == []

@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_resolve_id_skips_group_type_children(mock_simbad):
    """Checks that a child whose otype is itself a group (e.g. a sub-cluster) is skipped.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
    """
    mock_simbad.query_objectids.return_value = None
    mock_simbad.query_hierarchy.return_value = FakeTable(
        [{"main_id": "Sub Cluster", "otype": "Cl*"}, {"main_id": "Star A", "otype": "*"}],
        colnames=["main_id", "otype"],
    )
    mock_simbad.query_tap.return_value = [{"input_name": "Star A", "alias": "Gaia DR3 111"}]

    with patch("builtins.print"):
        result = resolve_id("Cluster", plot=False)

    assert result == [111]

@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_resolve_id_dedupes_duplicate_gaia_ids(mock_simbad):
    """Checks that the same Gaia ID reached via two different SIMBAD names appears only once.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
    """
    mock_simbad.query_objectids.return_value = None
    mock_simbad.query_hierarchy.return_value = FakeTable(
        [{"main_id": "Star A", "otype": "*"}, {"main_id": "Star A Alias", "otype": "*"}],
        colnames=["main_id", "otype"],
    )
    mock_simbad.query_tap.return_value = [
        {"input_name": "Star A", "alias": "Gaia DR3 111"},
        {"input_name": "Star A Alias", "alias": "Gaia DR3 111"},
    ]

    with patch("builtins.print"):
        result = resolve_id("Cluster", plot=False)

    assert result == [111]

@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_resolve_id_passes_membership_certainty_criteria(mock_simbad):
    """Checks that min_membership_certainty is translated into the SIMBAD hierarchy criteria string.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
    """
    mock_simbad.query_objectids.return_value = None
    mock_simbad.query_hierarchy.return_value = None

    with patch("builtins.print"):
        resolve_id("Cluster", plot=False, min_membership_certainty=90)

    _, kwargs = mock_simbad.query_hierarchy.call_args
    assert kwargs["criteria"] == "h_link.membership >= 90"

@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_resolve_id_membership_certainty_none_disables_filter(mock_simbad):
    """Checks that min_membership_certainty=None sends no criteria filter to SIMBAD.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
    """
    mock_simbad.query_objectids.return_value = None
    mock_simbad.query_hierarchy.return_value = None

    with patch("builtins.print"):
        resolve_id("Cluster", plot=False, min_membership_certainty=None)

    _, kwargs = mock_simbad.query_hierarchy.call_args
    assert kwargs["criteria"] is None

@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_resolve_id_radius_filter_drops_out_of_range_children(mock_simbad):
    """Checks that center_ra/center_dec/radius_deg drops children outside that radius.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
    """
    mock_simbad.query_objectids.return_value = None
    mock_simbad.query_hierarchy.return_value = FakeTable(
        [{"main_id": "Star A", "otype": "*"}, {"main_id": "Star B", "otype": "*"}],
        colnames=["main_id", "otype"],
    )
    mock_simbad.query_tap.return_value = [
        {"input_name": "Star A", "alias": "Gaia DR3 111"},
        {"input_name": "Star B", "alias": "Gaia DR3 222"},
    ]
    mock_simbad.query_objects.return_value = FakeTable(
        [{"main_id": "Star A", "ra": 10.0, "dec": 20.0}, {"main_id": "Star B", "ra": 50.0, "dec": 60.0}],
        colnames=["main_id", "ra", "dec"],
    )

    with patch("builtins.print"):
        result = resolve_id("Cluster", plot=False, center_ra=10.0, center_dec=20.0, radius_deg=1.0)

    assert result == [111]

@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_resolve_id_distance_soft_filter_drops_out_of_range_children(mock_simbad):
    """Checks that dist_min/dist_max drops a child whose implied distance is out of range.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
    """
    mock_simbad.query_objectids.return_value = None
    mock_simbad.query_hierarchy.return_value = FakeTable(
        [{"main_id": "Star A", "otype": "*"}, {"main_id": "Star B", "otype": "*"}],
        colnames=["main_id", "otype"],
    )
    mock_simbad.query_tap.return_value = [
        {"input_name": "Star A", "alias": "Gaia DR3 111"},
        {"input_name": "Star B", "alias": "Gaia DR3 222"},
    ]
    mock_simbad.query_objects.return_value = FakeTable(
        [{"main_id": "Star A", "plx_value": 100.0}, {"main_id": "Star B", "plx_value": 1.0}],
        colnames=["main_id", "plx_value"],
    )

    with patch("builtins.print"):
        result = resolve_id("Cluster", plot=False, dist_min=1, dist_max=50)

    assert result == [111]

@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_resolve_id_distance_soft_filter_keeps_missing_data(mock_simbad):
    """Checks that a child with no parallax on file is kept, not dropped, by the distance filter.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
    """
    mock_simbad.query_objectids.return_value = None
    mock_simbad.query_hierarchy.return_value = FakeTable(
        [{"main_id": "Star A", "otype": "*"}], colnames=["main_id", "otype"],
    )
    mock_simbad.query_tap.return_value = [{"input_name": "Star A", "alias": "Gaia DR3 111"}]
    mock_simbad.query_objects.return_value = FakeTable(
        [{"main_id": "Star A", "plx_value": None}], colnames=["main_id", "plx_value"],
    )

    with patch("builtins.print"):
        result = resolve_id("Cluster", plot=False, dist_min=1, dist_max=50)

    assert result == [111]

@patch("gaiadr3_analysis.gaia_input.sanity_check_star")
@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_resolve_id_calls_sanity_check_when_requested(mock_simbad, mock_sanity):
    """Checks that sanity_check_star is called with the resolved Gaia ID and release when sanity_check=True.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
        mock_sanity: Mocked sanity_check_star function.
    """
    mock_simbad.query_objectids.return_value = None
    with patch("builtins.print"):
        result = resolve_id(111, plot=False, sanity_check=True)
    assert result == 111
    mock_sanity.assert_called_once_with(111, release="dr3")

@patch("gaiadr3_analysis.gaia_input.save_dataframe_csv")
@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_resolve_id_saves_csv_when_requested(mock_simbad, mock_save):
    """Checks that save_csv writes a one-row CSV with the resolved star's info.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
        mock_save: Mocked save_dataframe_csv function.
    """
    mock_simbad.query_objectids.return_value = None
    with patch("builtins.print"):
        resolve_id(111, plot=False, save_csv=True, csv_file_name="my_star")

    mock_save.assert_called_once()
    df_arg, name_arg, default_arg = mock_save.call_args[0]
    assert df_arg.iloc[0]["gaia_id"] == 111
    assert name_arg == "my_star"

# find_cluster
def test_find_cluster_raises_value_error_without_position():
    """Checks that find_cluster raises ValueError when no position is given at all."""
    with pytest.raises(ValueError):
        find_cluster()

@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_find_cluster_returns_empty_df_when_no_region_results(mock_simbad):
    """Checks that find_cluster returns an empty DataFrame when query_region finds nothing nearby.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
    """
    mock_simbad.query_region.return_value = None
    with patch("builtins.print"):
        result = find_cluster(ra=10.0, dec=20.0)
    assert isinstance(result, pd.DataFrame)
    assert result.empty

@patch("gaiadr3_analysis.gaia_input.resolve_parents_batch")
@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_find_cluster_returns_empty_df_when_no_parent_resolves(mock_simbad, mock_parents):
    """Checks that find_cluster returns an empty DataFrame when no found star has a resolvable parent.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
        mock_parents: Mocked resolve_parents_batch function.
    """
    mock_simbad.query_region.return_value = FakeTable(
        [{"main_id": "Star A", "ra": 10.0, "dec": 20.0, "plx_value": 10.0}],
        colnames=["main_id", "ra", "dec", "plx_value"],
    )
    mock_parents.return_value = {"Star A": None}

    with patch("builtins.print"):
        result = find_cluster(ra=10.0, dec=20.0)

    assert result.empty

@patch("gaiadr3_analysis.gaia_input.resolve_children_batch")
@patch("gaiadr3_analysis.gaia_input.resolve_parents_batch")
@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_find_cluster_groups_members_by_parent(mock_simbad, mock_parents, mock_children):
    """Checks that stars sharing the same SIMBAD parent are grouped into one candidate cluster.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
        mock_parents: Mocked resolve_parents_batch function.
        mock_children: Mocked resolve_children_batch function (used for the cluster's common name).
    """
    mock_simbad.query_region.return_value = FakeTable(
        [
            {"main_id": "Star A", "ra": 10.0, "dec": 20.0, "plx_value": 10.0},
            {"main_id": "Star B", "ra": 10.1, "dec": 20.1, "plx_value": 10.0},
        ],
        colnames=["main_id", "ra", "dec", "plx_value"],
    )
    mock_parents.return_value = {
        "Star A": ("Cluster X", "OpC", 90.0),
        "Star B": ("Cluster X", "OpC", 90.0),
    }
    mock_children.return_value = {"Cluster X": {"gaia_id": None, "common_name": "Nice Cluster", "binaries": []}}

    with patch("builtins.print"):
        result = find_cluster(ra=10.0, dec=20.0)

    assert len(result) == 1
    assert result.iloc[0]["name"] == "Cluster X"
    assert result.iloc[0]["member_count"] == 2
    assert result.iloc[0]["common_name"] == "Nice Cluster"

@patch("gaiadr3_analysis.gaia_input.resolve_children_batch")
@patch("gaiadr3_analysis.gaia_input.resolve_parents_batch")
@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_find_cluster_membership_certainty_excludes_low_and_missing_scores(mock_simbad, mock_parents, mock_children):
    """Checks that min_membership_certainty excludes low-confidence AND missing-confidence member links.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
        mock_parents: Mocked resolve_parents_batch function.
        mock_children: Mocked resolve_children_batch function.
    """
    mock_simbad.query_region.return_value = FakeTable(
        [
            {"main_id": "Star A", "ra": 10.0, "dec": 20.0, "plx_value": 10.0},
            {"main_id": "Star B", "ra": 10.0, "dec": 20.0, "plx_value": 10.0},
            {"main_id": "Star C", "ra": 10.0, "dec": 20.0, "plx_value": 10.0},
        ],
        colnames=["main_id", "ra", "dec", "plx_value"],
    )
    mock_parents.return_value = {
        "Star A": ("Cluster X", "OpC", 90.0),
        "Star B": ("Cluster X", "OpC", 50.0),
        "Star C": ("Cluster X", "OpC", None),
    }
    mock_children.return_value = {}

    with patch("builtins.print"):
        result = find_cluster(ra=10.0, dec=20.0, min_membership_certainty=80)

    assert result.iloc[0]["member_count"] == 1

@patch("gaiadr3_analysis.gaia_input.resolve_children_batch")
@patch("gaiadr3_analysis.gaia_input.resolve_parents_batch")
@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_find_cluster_distance_filter_excludes_out_of_range_candidate(mock_simbad, mock_parents, mock_children):
    """Checks that dist_min/dist_max excludes a candidate whose implied distance is out of range.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
        mock_parents: Mocked resolve_parents_batch function.
        mock_children: Mocked resolve_children_batch function.
    """
    mock_simbad.query_region.return_value = FakeTable(
        [
            {"main_id": "Star A", "ra": 10.0, "dec": 20.0, "plx_value": 100.0},
            {"main_id": "Star B", "ra": 50.0, "dec": 60.0, "plx_value": 1.0},
        ],
        colnames=["main_id", "ra", "dec", "plx_value"],
    )
    mock_parents.return_value = {
        "Star A": ("Cluster Near", "OpC", 90.0),
        "Star B": ("Cluster Far", "OpC", 90.0),
    }
    mock_children.return_value = {}

    with patch("builtins.print"):
        result = find_cluster(ra=10.0, dec=20.0, dist_min=1, dist_max=50)

    assert list(result["name"]) == ["Cluster Near"]

@patch("gaiadr3_analysis.gaia_input.resolve_children_batch")
@patch("gaiadr3_analysis.gaia_input.resolve_parents_batch")
@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_find_cluster_ranks_by_member_count(mock_simbad, mock_parents, mock_children):
    """Checks that candidates are ranked by member_count, not by angular separation alone.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
        mock_parents: Mocked resolve_parents_batch function.
        mock_children: Mocked resolve_children_batch function.
    """
    mock_simbad.query_region.return_value = FakeTable(
        [
            {"main_id": "Star A", "ra": 10.0, "dec": 20.0, "plx_value": None},
            {"main_id": "Star B", "ra": 10.0, "dec": 20.0, "plx_value": None},
            {"main_id": "Star C", "ra": 10.0, "dec": 20.0, "plx_value": None},
            {"main_id": "Star D", "ra": 10.0, "dec": 20.0, "plx_value": None},
        ],
        colnames=["main_id", "ra", "dec", "plx_value"],
    )
    mock_parents.return_value = {
        "Star A": ("Big Cluster", "OpC", 90.0),
        "Star B": ("Big Cluster", "OpC", 90.0),
        "Star C": ("Big Cluster", "OpC", 90.0),
        "Star D": ("Small Cluster", "OpC", 90.0),
    }
    mock_children.return_value = {}

    with patch("builtins.print"):
        result = find_cluster(ra=10.0, dec=20.0)

    assert result.iloc[0]["name"] == "Big Cluster"
    assert result.iloc[0]["member_count"] == 3
    assert result.iloc[1]["name"] == "Small Cluster"

@patch("gaiadr3_analysis.gaia_input.resolve_children_batch")
@patch("gaiadr3_analysis.gaia_input.resolve_parents_batch")
@patch("gaiadr3_analysis.gaia_input.Simbad")
def test_find_cluster_respects_top_n(mock_simbad, mock_parents, mock_children):
    """Checks that find_cluster returns at most top_n candidates.

    Args:
        mock_simbad: Mocked Simbad class from astroquery.
        mock_parents: Mocked resolve_parents_batch function.
        mock_children: Mocked resolve_children_batch function.
    """
    rows = [{"main_id": f"Star {i}", "ra": 10.0, "dec": 20.0, "plx_value": None} for i in range(3)]
    mock_simbad.query_region.return_value = FakeTable(rows, colnames=["main_id", "ra", "dec", "plx_value"])
    mock_parents.return_value = {f"Star {i}": (f"Cluster {i}", "OpC", 90.0) for i in range(3)}
    mock_children.return_value = {}

    with patch("builtins.print"):
        result = find_cluster(ra=10.0, dec=20.0, top_n=2)

    assert len(result) == 2

# query_by_adql
def test_query_by_adql_raises_value_error_without_query_or_identifier():
    """Checks that query_by_adql raises ValueError when neither adql_query nor identifier is given."""
    with pytest.raises(ValueError):
        query_by_adql()

@patch("gaiadr3_analysis.gaia_input.Gaia")
def test_query_by_adql_runs_raw_query(mock_gaia):
    """Checks that query_by_adql runs a provided ADQL query and returns a DataFrame.

    Args:
        mock_gaia: Mocked Gaia class from astroquery.
    """
    mock_job = MagicMock()
    mock_job.get_results.return_value.to_pandas.return_value = pd.DataFrame({"source_id": [1, 2]})
    mock_gaia.launch_job.return_value = mock_job

    result = query_by_adql("SELECT TOP 2 source_id FROM gaiadr3.gaia_source")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2

@patch("gaiadr3_analysis.gaia_input.resolve_id")
@patch("gaiadr3_analysis.gaia_input.Gaia")
def test_query_by_adql_builds_default_query_from_identifier(mock_gaia, mock_resolve):
    """Checks that query_by_adql builds a default SELECT * query when only an identifier is given.

    Args:
        mock_gaia: Mocked Gaia class from astroquery.
        mock_resolve: Mocked resolve_id function.
    """
    mock_resolve.return_value = 123456789
    mock_job = MagicMock()
    mock_job.get_results.return_value.to_pandas.return_value = pd.DataFrame({"source_id": [123456789]})
    mock_gaia.launch_job.return_value = mock_job

    query_by_adql(identifier="HD 209458")
    called_query = mock_gaia.launch_job.call_args[0][0]
    assert "IN (123456789)" in called_query

@patch("gaiadr3_analysis.gaia_input.resolve_id")
def test_query_by_adql_returns_none_for_unresolved_identifier(mock_resolve):
    """Checks that query_by_adql returns None and doesn't run a query if the identifier can't be resolved.

    Args:
        mock_resolve: Mocked resolve_id function.
    """
    mock_resolve.return_value = None
    with patch("builtins.print"):
        result = query_by_adql(identifier="Unresolvable Star")
    assert result is None

@patch("gaiadr3_analysis.gaia_input.resolve_id")
@patch("gaiadr3_analysis.gaia_input.Gaia")
def test_query_by_adql_multiple_ids_joined_with_in_clause(mock_gaia, mock_resolve):
    """Checks that a list of resolved IDs is substituted as a comma-separated IN clause.

    Args:
        mock_gaia: Mocked Gaia class from astroquery.
        mock_resolve: Mocked resolve_id function.
    """
    mock_resolve.return_value = [111, 222, 333]
    mock_job = MagicMock()
    mock_job.get_results.return_value.to_pandas.return_value = pd.DataFrame({"source_id": [111, 222, 333]})
    mock_gaia.launch_job.return_value = mock_job

    query_by_adql(identifier="Some Cluster")

    called_query = mock_gaia.launch_job.call_args[0][0]
    assert "IN (111,222,333)" in called_query

@patch("gaiadr3_analysis.gaia_input.Gaia")
def test_query_by_adql_saves_to_default_filename_when_none_given(mock_gaia):
    """Checks that query_by_adql falls back to a default filename when saving without an identifier.

    Args:
        mock_gaia: Mocked Gaia class from astroquery.
    """
    mock_job = MagicMock()
    mock_job.get_results.return_value.to_pandas.return_value = pd.DataFrame({"source_id": [1]})
    mock_gaia.launch_job.return_value = mock_job

    with patch("pandas.DataFrame.to_csv") as mock_to_csv:
        query_by_adql("SELECT TOP 1 source_id FROM gaiadr3.gaia_source", save_file=True)

    mock_to_csv.assert_called_once_with("gaia_query.csv")

@patch("gaiadr3_analysis.gaia_input.resolve_id")
@patch("gaiadr3_analysis.gaia_input.Gaia")
def test_query_by_adql_dr4_uses_dr4_source_table(mock_gaia, mock_resolve):
    """Checks that the auto-built default query targets the DR4 source table when release='dr4'.

    Args:
        mock_gaia: Mocked Gaia class from astroquery.
        mock_resolve: Mocked resolve_id function.
    """
    mock_resolve.return_value = 111
    mock_job = MagicMock()
    mock_job.get_results.return_value.to_pandas.return_value = pd.DataFrame({"source_id": [111]})
    mock_gaia.launch_job.return_value = mock_job

    query_by_adql(identifier="Some Star", release="dr4")

    called_query = mock_gaia.launch_job.call_args[0][0]
    assert "gaiadr4.gaia_source" in called_query

# query_by_datalink
def test_query_by_datalink_raises_type_error_for_bad_folder_name():
    """Checks that query_by_datalink raises TypeError when save_file is True and folder_name isn't a string."""
    with pytest.raises(TypeError):
        query_by_datalink(123456789, save_file=True, folder_name=None)

@patch("gaiadr3_analysis.gaia_input.resolve_id")
def test_query_by_datalink_returns_empty_dict_when_nothing_resolves(mock_resolve):
    """Checks that query_by_datalink returns an empty dict if no IDs could be resolved.

    Args:
        mock_resolve: Mocked resolve_id function.
    """
    mock_resolve.return_value = None

    with patch("builtins.print"):
        result = query_by_datalink(["Unresolvable Star"])

    assert result == {}

@patch("gaiadr3_analysis.gaia_input.resolve_id")
@patch("gaiadr3_analysis.gaia_input.Gaia")
def test_query_by_datalink_flattens_list_results_from_resolve_id(mock_gaia, mock_resolve):
    """Checks that a resolve_id call returning a list of children IDs is flattened into the query.

    Args:
        mock_gaia: Mocked Gaia class from astroquery.
        mock_resolve: Mocked resolve_id function.
    """
    mock_resolve.return_value = [111, 222]
    mock_table = MagicMock()
    mock_table.to_table.return_value.to_pandas.return_value = pd.DataFrame({"flux": [1.0]})
    mock_gaia.load_data.return_value = {
        f"{KEY_PREFIX}111.xml": [mock_table],
        f"{KEY_PREFIX}222.xml": [mock_table],
    }

    result = query_by_datalink("Some Cluster")

    called_ids = mock_gaia.load_data.call_args.kwargs["ids"]
    assert called_ids == [111, 222]
    assert set(result.keys()) == {111, 222}

@patch("gaiadr3_analysis.gaia_input.resolve_id")
@patch("gaiadr3_analysis.gaia_input.Gaia")
def test_query_by_datalink_reports_unretrieved_ids(mock_gaia, mock_resolve):
    """Checks that IDs with no epoch photometry data are reported as not retrieved.

    Args:
        mock_gaia: Mocked Gaia class from astroquery.
        mock_resolve: Mocked resolve_id function.
    """
    mock_resolve.side_effect = [111, 222]
    mock_table = MagicMock()
    mock_table.to_table.return_value.to_pandas.return_value = pd.DataFrame({"flux": [1.0]})
    mock_gaia.load_data.return_value = {f"{KEY_PREFIX}111.xml": [mock_table]}

    with patch("builtins.print") as mock_print:
        result = query_by_datalink([111, 222])

    assert 111 in result
    assert 222 not in result
    printed = " ".join(str(call.args) for call in mock_print.call_args_list)
    assert "222" in printed

@patch("gaiadr3_analysis.gaia_input.resolve_id")
@patch("gaiadr3_analysis.gaia_input.Gaia")
def test_query_by_datalink_saves_csv_per_star(mock_gaia, mock_resolve):
    """Checks that query_by_datalink writes one CSV file per resolved star when save_file is True.

    Args:
        mock_gaia: Mocked Gaia class from astroquery.
        mock_resolve: Mocked resolve_id function.
    """
    mock_resolve.return_value = 111
    mock_table = MagicMock()
    mock_table.to_table.return_value.to_pandas.return_value = pd.DataFrame({"flux": [1.0]})
    mock_gaia.load_data.return_value = {f"{KEY_PREFIX}111.xml": [mock_table]}

    with patch("pandas.DataFrame.to_csv") as mock_to_csv:
        query_by_datalink(111, save_file=True, folder_name="output")

    mock_to_csv.assert_called_once_with("output/111.csv")

@patch("gaiadr3_analysis.gaia_input.resolve_id")
@patch("gaiadr3_analysis.gaia_input.Gaia")
def test_query_by_datalink_dr4_uses_dr4_release_string(mock_gaia, mock_resolve):
    """Checks that query_by_datalink passes the Gaia DR4 release string to Gaia.load_data when release='dr4'.

    Args:
        mock_gaia: Mocked Gaia class from astroquery.
        mock_resolve: Mocked resolve_id function.
    """
    mock_resolve.return_value = 111
    mock_table = MagicMock()
    mock_table.to_table.return_value.to_pandas.return_value = pd.DataFrame({"flux": [1.0]})
    mock_gaia.load_data.return_value = {f"{KEY_PREFIX}111.xml": [mock_table]}

    query_by_datalink(111, release="dr4")

    assert mock_gaia.load_data.call_args.kwargs["data_release"] == "Gaia DR4"

# find_star
def test_find_star_raises_value_error_without_position():
    """Checks that find_star raises ValueError when no position is given at all."""
    with pytest.raises(ValueError):
        find_star()

@patch("gaiadr3_analysis.gaia_input.fetch_common_names")
@patch("gaiadr3_analysis.gaia_input.query_by_adql")
def test_find_star_runs_with_string_ra_dec(mock_query, mock_names):
    """Checks that find_star converts sexagesimal ra/dec strings and runs a query.

    Args:
        mock_query: Mocked query_by_adql function.
        mock_names: Mocked fetch_common_names function.
    """
    mock_names.return_value = {}
    mock_query.return_value = pd.DataFrame({"source_id": [1]})

    result = find_star(ra="06h45m08.9s", dec="-16d42m58s")

    assert mock_query.called
    assert isinstance(result, pd.DataFrame)

@patch("gaiadr3_analysis.gaia_input.fetch_common_names")
@patch("gaiadr3_analysis.gaia_input.query_by_adql")
def test_find_star_runs_with_skycoord(mock_query, mock_names):
    """Checks that find_star accepts a SkyCoord object directly.

    Args:
        mock_query: Mocked query_by_adql function.
        mock_names: Mocked fetch_common_names function.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    mock_names.return_value = {}
    mock_query.return_value = pd.DataFrame({"source_id": [1]})
    coord = SkyCoord(ra=101.287 * u.deg, dec=-16.716 * u.deg)

    result = find_star(coordinates=coord)

    assert mock_query.called
    assert isinstance(result, pd.DataFrame)

@patch("gaiadr3_analysis.gaia_input.fetch_common_names")
@patch("gaiadr3_analysis.gaia_input.query_by_adql")
def test_find_star_passes_save_csv_through(mock_query, mock_names):
    """Checks that find_star forwards save_csv/csv_file_name to query_by_adql's save_file/file_name.

    Args:
        mock_query: Mocked query_by_adql function.
        mock_names: Mocked fetch_common_names function.
    """
    mock_names.return_value = {}
    mock_query.return_value = pd.DataFrame({"source_id": [1]})

    find_star(ra="06h45m08.9s", dec="-16d42m58s", save_csv=True, csv_file_name="my_star")

    _, kwargs = mock_query.call_args
    assert kwargs.get("save_file") is True
    assert kwargs.get("file_name") == "my_star"

@patch("gaiadr3_analysis.gaia_input.fetch_common_names")
@patch("gaiadr3_analysis.gaia_input.query_by_adql")
def test_find_star_builds_hard_magnitude_filter(mock_query, mock_names):
    """Checks that mag_min/mag_max are built directly into the ADQL query (a hard filter).

    Args:
        mock_query: Mocked query_by_adql function.
        mock_names: Mocked fetch_common_names function.
    """
    mock_names.return_value = {}
    mock_query.return_value = pd.DataFrame({"source_id": [1]})

    find_star(ra="06h45m08.9s", dec="-16d42m58s", mag_min=5, mag_max=15)

    called_query = mock_query.call_args[0][0]
    assert "phot_g_mean_mag >= 5" in called_query
    assert "phot_g_mean_mag <= 15" in called_query

@patch("gaiadr3_analysis.gaia_input.fetch_common_names")
@patch("gaiadr3_analysis.gaia_input.query_by_adql")
def test_find_star_distance_soft_filter_excludes_out_of_range(mock_query, mock_names):
    """Checks that dist_min/dist_max is applied after the query (a soft filter), not in the SQL.

    Args:
        mock_query: Mocked query_by_adql function.
        mock_names: Mocked fetch_common_names function.
    """
    mock_names.return_value = {}
    mock_query.return_value = pd.DataFrame({
        "source_id": [1, 2],
        "parallax": [100.0, 1.0],
    })

    result = find_star(ra="06h45m08.9s", dec="-16d42m58s", dist_min=1, dist_max=50)

    assert list(result["source_id"]) == [1]
    called_query = mock_query.call_args[0][0]
    assert "parallax" not in called_query

@patch("gaiadr3_analysis.gaia_input.fetch_common_names")
@patch("gaiadr3_analysis.gaia_input.query_by_adql")
def test_find_star_distance_filter_keeps_missing_parallax(mock_query, mock_names):
    """Checks that a source with no parallax on file is kept, not dropped, by the distance filter.

    Args:
        mock_query: Mocked query_by_adql function.
        mock_names: Mocked fetch_common_names function.
    """
    mock_names.return_value = {}
    mock_query.return_value = pd.DataFrame({"source_id": [1], "parallax": [None]})

    result = find_star(ra="06h45m08.9s", dec="-16d42m58s", dist_min=1, dist_max=50)

    assert list(result["source_id"]) == [1]

@patch("gaiadr3_analysis.gaia_input.fetch_common_names")
@patch("gaiadr3_analysis.gaia_input.query_by_adql")
def test_find_star_adds_common_name_column_after_source_id(mock_query, mock_names):
    """Checks that the common_name column is populated and placed right after source_id.

    Args:
        mock_query: Mocked query_by_adql function.
        mock_names: Mocked fetch_common_names function.
    """
    mock_names.return_value = {1: "Proxima Centauri"}
    mock_query.return_value = pd.DataFrame({"source_id": [1], "ra": [1.0], "dec": [2.0]})

    result = find_star(ra="06h45m08.9s", dec="-16d42m58s", plot=False)

    assert result.iloc[0]["common_name"] == "Proxima Centauri"
    cols = list(result.columns)
    assert cols.index("common_name") == cols.index("source_id") + 1

@patch("gaiadr3_analysis.gaia_input.fetch_common_names")
@patch("gaiadr3_analysis.gaia_input.query_by_adql")
def test_find_star_dr4_uses_dr4_source_table(mock_query, mock_names):
    """Checks that find_star's cone-search query targets the DR4 source table when release='dr4'.

    Args:
        mock_query: Mocked query_by_adql function.
        mock_names: Mocked fetch_common_names function.
    """
    mock_names.return_value = {}
    mock_query.return_value = pd.DataFrame({"source_id": [1]})

    find_star(ra="06h45m08.9s", dec="-16d42m58s", release="dr4", plot=False)

    called_query = mock_query.call_args[0][0]
    assert "gaiadr4.gaia_source" in called_query

# gaia_login_prompt
def test_gaia_login_prompt_skips_login_on_no():
    """Checks that gaia_login_prompt does not attempt login when the user declines."""
    with patch("builtins.input", return_value="n"), patch("gaiadr3_analysis.gaia_input.Gaia") as mock_gaia:
        gaia_login_prompt()
    mock_gaia.login.assert_not_called()

@patch("gaiadr3_analysis.gaia_input.Gaia")
def test_gaia_login_prompt_logs_in_on_yes(mock_gaia):
    """Checks that gaia_login_prompt calls Gaia.login with the entered credentials.

    Args:
        mock_gaia: Mocked Gaia class from astroquery.
    """
    with patch("builtins.input", side_effect=["y", "test_user", "test_pass"]), patch("builtins.print"):
        gaia_login_prompt()

    mock_gaia.login.assert_called_once_with(user="test_user", password="test_pass")

@patch("gaiadr3_analysis.gaia_input.Gaia")
def test_gaia_login_prompt_handles_login_failure(mock_gaia):
    """Checks that gaia_login_prompt catches and reports a failed login attempt.

    Args:
        mock_gaia: Mocked Gaia class from astroquery.
    """
    mock_gaia.login.side_effect = Exception("bad credentials")

    with patch("builtins.input", side_effect=["y", "test_user", "wrong_pass"]), patch("builtins.print") as mock_print:
        gaia_login_prompt()

    assert any("Login failed" in str(call.args) for call in mock_print.call_args_list)

# get_dataframe
@patch("gaiadr3_analysis.gaia_input.query_by_adql")
@patch("gaiadr3_analysis.gaia_input.gaia_login_prompt")
def test_get_dataframe_adql_identifier_path(mock_login, mock_query):
    """Checks that choosing the ADQL/identifier menu path calls query_by_adql with the right args.

    Args:
        mock_login: Mocked gaia_login_prompt function.
        mock_query: Mocked query_by_adql function.
    """
    mock_query.return_value = pd.DataFrame({"source_id": [1]})
    inputs = iter(["1", "y", "HD 209458", "n"])

    with patch("builtins.input", lambda *_: next(inputs)):
        result = get_dataframe()

    mock_query.assert_called_once_with(None, identifier="HD 209458")
    assert isinstance(result, pd.DataFrame)

@patch("gaiadr3_analysis.gaia_input.load_csv")
@patch("gaiadr3_analysis.gaia_input.gaia_login_prompt")
def test_get_dataframe_csv_path(mock_login, mock_load_csv):
    """Checks that choosing the CSV menu option calls load_csv with the given path.

    Args:
        mock_login: Mocked gaia_login_prompt function.
        mock_load_csv: Mocked load_csv function.
    """
    mock_load_csv.return_value = pd.DataFrame({"source_id": [1]})
    inputs = iter(["2", "data/my_stars.csv"])

    with patch("builtins.input", lambda *_: next(inputs)):
        result = get_dataframe()

    mock_load_csv.assert_called_once_with("data/my_stars.csv")
    assert isinstance(result, pd.DataFrame)

@patch("gaiadr3_analysis.gaia_input.query_by_datalink")
@patch("gaiadr3_analysis.gaia_input.gaia_login_prompt")
def test_get_dataframe_datalink_path(mock_login, mock_query_dl):
    """Checks that choosing the Datalink menu option calls query_by_datalink with parsed IDs.

    Args:
        mock_login: Mocked gaia_login_prompt function.
        mock_query_dl: Mocked query_by_datalink function.
    """
    mock_query_dl.return_value = {111: pd.DataFrame({"flux": [1.0]})}
    inputs = iter(["3", "111, 222", "n"])

    with patch("builtins.input", lambda *_: next(inputs)):
        result = get_dataframe()

    mock_query_dl.assert_called_once_with(["111", "222"])
    assert result is mock_query_dl.return_value

@patch("gaiadr3_analysis.gaia_input.gaia_login_prompt")
def test_get_dataframe_invalid_choice_returns_none(mock_login):
    """Checks that an invalid menu choice prints an error and returns None.

    Args:
        mock_login: Mocked gaia_login_prompt function.
    """
    with patch("builtins.input", return_value="9"), patch("builtins.print"):
        result = get_dataframe()

    assert result is None