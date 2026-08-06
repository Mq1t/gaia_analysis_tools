"""gaia_input.py

Load GAIA data into a pandas DataFrame via three input methods:
    1. ADQL query
    2. CSV file upload
    3. Datalink

Usage:
    $ python gaia_input.py
"""

import pandas as pd
import statistics
import matplotlib.pyplot as plt
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u
from .constants import SPTYPE_TEFF_RANGES

GAIA_RELEASE_TABLES = {
    "dr3": {
        "source_table": "gaiadr3.gaia_source",
        "ap_table": "gaiadr3.astrophysical_parameters",
        "designation_prefix": "Gaia DR3",
        "datalink_release": "Gaia DR3",
    },
    "dr4": {
        # DR4 is unreleased and may need updating once DR4's final data model is published.
        "source_table": "gaiadr4.gaia_source",
        "ap_table": "gaiadr4.ap_xp",
        "designation_prefix": "Gaia DR4",
        "datalink_release": "Gaia DR4",
    },
}

def gaia_designation(gaia_id, release="dr3"):
    """Build a Gaia designation string for a source ID, for the given release.

    Args:
        gaia_id (int): Gaia source ID.
        release (str, optional): Gaia data release to build the designation for, "dr3" or "dr4". Defaults to "dr3".

    Returns:
        str: e.g. "Gaia DR3 123456789" or "Gaia DR4 123456789".
    """
    return f"{GAIA_RELEASE_TABLES[release]['designation_prefix']} {gaia_id}"

def is_value_masked(value):
    """Check whether a SIMBAD table value is masked (i.e. missing/not on file).

    SIMBAD results come back as astropy Tables, where a cell with no data on file is a masked value rather than None. 

    Args:
        value: A value pulled from an astropy table row.

    Returns:
        bool: True if the value is masked (no data), False otherwise.
    """
    return hasattr(value, "mask") and value.mask

def query_simbad_by_gaia_ids(gaia_ids, release="dr3", error_context="SIMBAD lookup"):
    """Query SIMBAD for one or more Gaia source IDs, by their release-specific designation.

    Args:
        gaia_ids (list[int]): Gaia source ID(s) to look up.
        release (str, optional): Gaia data release the IDs belong to, "dr3" or "dr4". Defaults to "dr3".
        error_context (str, optional): Used in the printed message if the query fails.

    Returns:
        astropy.table.Table or None: SIMBAD's result table (one row per gaia_ids entry, in the same order),
        or None if the query failed.
    """
    try:
        query_names = [gaia_designation(gid, release) for gid in gaia_ids]
        return Simbad.query_objects(query_names)
    except Exception as e:
        print(f"  ({error_context} failed: {e})")
        return None

def save_dataframe_csv(df, file_name, default_name):
    """Save a DataFrame to a CSV file, formatting the file name consistently.

    Args:
        df (pandas.DataFrame): Data to save.
        file_name (str or None): Desired file name. Falls back to default_name if None.
        default_name (str): File name to use if file_name is not given.

    Returns:
        str: The final formatted file name that was saved to.
    """
    formatted_name = (file_name or default_name).replace(" ", "_")
    if not formatted_name.lower().endswith(".csv"):
        formatted_name = f"{formatted_name}.csv"
    df.to_csv(formatted_name, index=False)
    print(f"  Saved to {formatted_name}")
    return formatted_name

def resolve_search_position(ra=None, dec=None, coordinates=None):
    """Resolve an RA/Dec position from either sexagesimal strings or a SkyCoord.

    Args:
        ra (str or float, optional): Right ascension, as a sexagesimal string (e.g. "10h21m00s") when
            paired with a string `dec`. Defaults to None.
        dec (str or float, optional): Declination, as a sexagesimal string (e.g. "+41d05m00s") when
            paired with a string `ra`. Defaults to None.
        coordinates (astropy.coordinates.SkyCoord, optional): Sky position to use if `ra`/`dec` are
            not given. Defaults to None.

    Returns:
        tuple[float, float]: (ra_deg, dec_deg).

    Raises:
        ValueError: If neither a valid ra/dec pair nor coordinates is given.
    """
    if isinstance(ra, str) and isinstance(dec, str):
        deg_coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
        ra = deg_coord.ra.deg
        dec = deg_coord.dec.deg
    elif (ra is None and dec is None) and coordinates is not None:
        ra = coordinates.ra.deg
        dec = coordinates.dec.deg

    if ra is None or dec is None:
        raise ValueError("Provide either (ra and dec) as strings, or coordinates as a SkyCoord.")

    return ra, dec

def plot_positions(points, title, save_plot=False, plot_file_name=None, center=None, radius_deg=None, default_file_name="plot", label_points=None, label_auto_max=15):
    """Plot RA/Dec positions on a matplotlib scatter plot.

    Args:
        points (list[tuple[str, float, float]]): (label, ra_deg, dec_deg) for each point to plot.
        title (str): Plot title.
        save_plot (bool, optional): Whether to save the plot to an image file. Defaults to False.
        plot_file_name (str, optional): Name or path for the saved plot image. Defaults to None.
        center (tuple[float, float], optional): (ra_deg, dec_deg) center of a search area to mark. Defaults to None.
        radius_deg (float, optional): Search radius in degrees, drawn as a dashed circle around center. Defaults to None.
        default_file_name (str, optional): File name to use if save_plot is True and plot_file_name is not given. 
        label_points (bool, optional): Whether to label each point with its name. Defaults to None
            (auto: labels are shown if there are label_auto_max points or fewer, hidden otherwise).
        label_auto_max (int, optional): Point count threshold used for the auto behavior above.
            Only relevant when label_points is None. Defaults to 15.
    """
    if not points:
        print("(nothing to plot - no coordinates available)")
        return

    show_labels = label_points if label_points is not None else len(points) <= label_auto_max

    try:
        fig, ax = plt.subplots(figsize=(7, 6))

        ras = [p[1] for p in points]
        decs = [p[2] for p in points]
        ax.scatter(ras, decs, c="tab:blue", s=40, zorder=3)

        if show_labels:
            for label, ra, dec in points:
                ax.annotate(label, (ra, dec), fontsize=8, xytext=(4, 4), textcoords="offset points")

        if center is not None and radius_deg is not None:
            circle = plt.Circle(center, radius_deg, fill=False, edgecolor="tab:red", linestyle="--", zorder=2)
            ax.add_patch(circle)
            ax.scatter([center[0]], [center[1]], marker="x", c="tab:red", s=60, zorder=4, label="search center")
            ax.legend(loc="upper right", fontsize=8)

        ax.set_xlabel("RA (deg)")
        ax.set_ylabel("Dec (deg)")
        ax.set_title(title)
        ax.invert_xaxis()
        plt.tight_layout()

        if save_plot:
            formatted_name = (plot_file_name or default_file_name).replace(" ", "_")
            if not formatted_name.lower().endswith((".png", ".jpg", ".jpeg", ".pdf", ".svg")):
                formatted_name = f"{formatted_name}.png"
            fig.savefig(formatted_name)
            print(f"  Plot saved to {formatted_name}")

        plt.show()
    except Exception as e:
        print(f"  (plotting skipped: {e})")

def strip_name_prefix(raw_name):
    """Strip a leading 'NAME ' prefix from a raw SIMBAD main_id, for cleaner display.

    Args:
        raw_name (str): Raw SIMBAD identifier/name string.

    Returns:
        str: The name with a leading 'NAME ' removed, if present.
    """
    raw_name = str(raw_name)
    return raw_name[5:] if raw_name.startswith("NAME ") else raw_name

def find_name_and_binaries(identifier_table):
    """Pull a common name and any binary/multiple star designations from a SIMBAD ID result.

    Args:
        identifier_table (astropy.table.Table): Result of Simbad.query_objectids() - a table of
            all known identifiers/aliases for one object.

    Returns:
        tuple[str or None, list[str]]: Common name (or None if not found), and a list of any
        binary/multiple star designations found.
    """
    common_name = None
    binaries = []
    for row in identifier_table["id"]:
        if row.startswith("NAME ") and common_name is None:
            common_name = strip_name_prefix(row)
        if row.startswith("** "):
            binaries.append(row)
    return common_name, binaries

def fetch_coordinates(gaia_ids, release="dr3"):
    """Look up RA/Dec (in degrees) for one or more Gaia source IDs, for plotting.

    Tries SIMBAD first. Any ID SIMBAD has no usable position for (no cross-match, or a masked/missing ra/dec) 
    is looked up directly in the Gaia archive instead. An ID that fails both lookups is not a valid Gaia source 
    for the given release, and is skipped with a warning.

    Args:
        gaia_ids (int or list[int]): Gaia source ID(s).
        release (str, optional): Gaia data release to look up, "dr3" or "dr4". Defaults to "dr3".

    Returns:
        list[tuple[str, float, float]]: (label, ra_deg, dec_deg) for each ID successfully resolved
        to a position. Label is the SIMBAD name if available, else the Gaia ID itself.
    """
    if isinstance(gaia_ids, int):
        gaia_ids = [gaia_ids]

    points = []
    if not gaia_ids:
        return points

    found_ids = set()

    result = query_simbad_by_gaia_ids(gaia_ids, release, error_context="coordinate lookup for plotting")

    if result is not None:
        for i, row in enumerate(result):
            gid = gaia_ids[i] if i < len(gaia_ids) else None
            if gid is None:
                continue
            try:
                label = strip_name_prefix(row["main_id"]) if "main_id" in result.colnames and row["main_id"] else str(gid)

                if "ra" in result.colnames and "dec" in result.colnames:
                    ra_val, dec_val = row["ra"], row["dec"]
                else:
                    ra_val, dec_val = row["RA"], row["DEC"]

                if ra_val is None or dec_val is None:
                    continue
                if is_value_masked(ra_val) or is_value_masked(dec_val):
                    continue

                if isinstance(ra_val, str):
                    coord = SkyCoord(ra=ra_val, dec=dec_val, unit=(u.hourangle, u.deg))
                else:
                    coord = SkyCoord(ra=ra_val * u.deg, dec=dec_val * u.deg)

                points.append((label, coord.ra.deg, coord.dec.deg))
                found_ids.add(gid)
            except Exception:
                continue

    # Fall back to the Gaia archive
    missing_ids = [gid for gid in gaia_ids if gid not in found_ids]
    if missing_ids:
        try:
            id_list = ",".join(str(g) for g in missing_ids)
            source_table = GAIA_RELEASE_TABLES[release]["source_table"]
            gaia_query = f"SELECT source_id, ra, dec FROM {source_table} WHERE source_id IN ({id_list})"
            job = Gaia.launch_job(gaia_query)
            gaia_df = job.get_results().to_pandas()
            for _, row in gaia_df.iterrows():
                gid = int(row["source_id"])
                points.append((str(gid), float(row["ra"]), float(row["dec"])))
                found_ids.add(gid)
        except Exception as e:
            print(f"  (Gaia archive fallback lookup for plotting failed: {e})")

    not_found = [gid for gid in gaia_ids if gid not in found_ids]
    if not_found:
        release_label = GAIA_RELEASE_TABLES[release]["designation_prefix"]
        print(f"  Warning: could not find coordinates for {release_label} ID(s) {not_found} - they may not be valid {release_label} source(s).")

    return points

def fetch_common_names(gaia_ids, batch_size=300, release="dr3"):
    """Look up a display name for one or more Gaia source IDs, via SIMBAD.

    Prefers a true name when SIMBAD has one on file. Otherwise falls back to the star's main SIMBAD identifier.

    Args:
        gaia_ids (list[int]): Gaia source ID(s).
        batch_size (int, optional): Max number of IDs per batched proper-name query. Defaults to 300.
        release (str, optional): Gaia data release the IDs belong to, "dr3" or "dr4". Note SIMBAD
            has no Gaia DR4 cross-matches yet (DR4 is unreleased), so with release="dr4" this will
            currently return no names for any ID. Defaults to "dr3".

    Returns:
        dict[int, str]: Maps each Gaia ID that SIMBAD has any name for to that name. IDs with no
        SIMBAD record at all are simply absent from the returned dict.
    """
    names = {}
    if not gaia_ids:
        return names

    designation_prefix = GAIA_RELEASE_TABLES[release]["designation_prefix"]

    # Fallback
    result = query_simbad_by_gaia_ids(gaia_ids, release, error_context="common name lookup")
    if result is not None:
        for i, row in enumerate(result):
            gid = gaia_ids[i] if i < len(gaia_ids) else None
            if gid is not None and "main_id" in result.colnames and row["main_id"]:
                names[gid] = strip_name_prefix(row["main_id"])

    # Preferred
    unique_ids = list(dict.fromkeys(gaia_ids))
    for i in range(0, len(unique_ids), batch_size):
        chunk = unique_ids[i:i + batch_size]
        id_list = ", ".join(f"'{gaia_designation(gid, release)}'" for gid in chunk)
        query = (
            "SELECT id_typed.id AS input_name, ident.id AS alias "
            "FROM ident AS id_typed JOIN ident USING(oidref) "
            f"WHERE id_typed.id IN ({id_list})"
        )
        try:
            chunk_result = Simbad.query_tap(query)
        except Exception as e:
            print(f"  (proper name lookup failed for a chunk of {len(chunk)} IDs: {e})")
            continue

        if chunk_result is None:
            continue

        for row in chunk_result:
            alias = str(row["alias"])
            if not alias.startswith("NAME "):
                continue
            try:
                gid = int(str(row["input_name"]).replace(designation_prefix + " ", ""))
            except ValueError:
                continue
            names[gid] = strip_name_prefix(alias)

    if release == "dr4" and not names:
        print(" Note: no common names found - SIMBAD has no Gaia DR4 cross-matches yet.")

    return names

def fetch_mag_and_parallax(gaia_ids, mag_band="V", release="dr3"):
    """Look up SIMBAD magnitude and parallax for one or more Gaia source IDs.

    Args:
        gaia_ids (list[int]): Gaia source ID(s).
        mag_band (str, optional): SIMBAD photometric band to fetch. Defaults to "V".
        release (str, optional): Gaia data release the IDs belong to, "dr3" or "dr4". Defaults to "dr3".

    Returns:
        dict[int, dict]: Maps each Gaia ID to a dict with keys 'mag' and 'plx_value' (float or None if not available).
    """
    results = {gid: {"mag": None, "plx_value": None} for gid in gaia_ids}
    if not gaia_ids:
        return results

    try:
        Simbad.add_votable_fields(f'flux({mag_band})', 'parallax')
    except Exception as e:
        print(f"  (could not request magnitude/parallax fields: {e})")
        return results

    result = query_simbad_by_gaia_ids(gaia_ids, release, error_context="magnitude/parallax lookup")
    if result is None:
        return results

    for i, row in enumerate(result):
        gid = gaia_ids[i] if i < len(gaia_ids) else None
        if gid is None:
            continue
        try:
            for mag_col in (f"flux_{mag_band}", f"FLUX_{mag_band}", mag_band.lower(), mag_band.upper()):
                if mag_col in result.colnames:
                    mag_val = row[mag_col]
                    if mag_val is not None and not is_value_masked(mag_val):
                        results[gid]["mag"] = float(mag_val)
                        break
            if "plx_value" in result.colnames:
                plx_val = row["plx_value"]
                if plx_val is not None and not is_value_masked(plx_val) and plx_val > 0:
                    results[gid]["plx_value"] = float(plx_val)
        except Exception:
            continue

    return results

def resolve_children_batch(child_names, batch_size=300, release="dr3"):
    """Resolve many SIMBAD names to Gaia IDs (plus common name/binaries) in batched queries.

    Args:
        child_names (list[str]): SIMBAD main_id names to resolve.
        batch_size (int, optional): Max number of names per batched SIMBAD query. Defaults to 300.
        release (str, optional): Gaia data release to resolve IDs against, "dr3" or "dr4". Defaults to "dr3".

    Returns:
        dict[str, dict]: Maps each input name to a dict with keys 'gaia_id' (int or None),
        'common_name' (str or None), and 'binaries' (list[str]).
    """
    results = {name: {"gaia_id": None, "common_name": None, "binaries": []} for name in child_names}
    if not child_names:
        return results

    designation_prefix = GAIA_RELEASE_TABLES[release]["designation_prefix"]
    unique_names = list(dict.fromkeys(child_names))

    for i in range(0, len(unique_names), batch_size):
        chunk = unique_names[i:i + batch_size]
        name_list = ", ".join("'{}'".format(n.replace("'", "''")) for n in chunk)
        query = (
            "SELECT id_typed.id AS input_name, ident.id AS alias "
            "FROM ident AS id_typed JOIN ident USING(oidref) "
            f"WHERE id_typed.id IN ({name_list})"
        )
        try:
            chunk_result = Simbad.query_tap(query)
        except Exception as e:
            print(f"  (batched resolution failed for a chunk of {len(chunk)} names: {e})")
            continue

        if chunk_result is None:
            continue

        for row in chunk_result:
            input_name = str(row["input_name"])
            alias = str(row["alias"])
            if input_name not in results:
                continue
            if alias.startswith(designation_prefix + " ") and results[input_name]["gaia_id"] is None:
                results[input_name]["gaia_id"] = int(alias.replace(designation_prefix + " ", ""))
            elif alias.startswith("NAME ") and results[input_name]["common_name"] is None:
                results[input_name]["common_name"] = strip_name_prefix(alias)
            elif alias.startswith("** "):
                results[input_name]["binaries"].append(alias)

    return results

def resolve_parents_batch(child_names, batch_size=300):
    """Find the SIMBAD hierarchy parent for many star names at once, in batched queries.

    Used to identify clusters by their members.

    Args:
        child_names (list[str]): SIMBAD main_id names of individual stars.
        batch_size (int, optional): Max number of names per batched SIMBAD query. Defaults to 300.

    Returns:
        dict[str, tuple[str, str, float or None] or None]: Maps each input name to
        (parent_name, parent_otype, membership_certainty), or None if no parent link was found.
        membership_certainty (0-100, set by the authors of the source paper) is None if not on file.
    """
    results = {name: None for name in child_names}
    if not child_names:
        return results

    unique_names = list(dict.fromkeys(child_names))

    for i in range(0, len(unique_names), batch_size):
        chunk = unique_names[i:i + batch_size]
        name_list = ", ".join("'{}'".format(n.replace("'", "''")) for n in chunk)
        query = (
            "SELECT child_ident.id AS child_name, parent_basic.main_id AS parent_name, "
            "parent_basic.otype AS parent_otype, h_link.membership AS membership_certainty "
            "FROM ident AS child_ident "
            "JOIN h_link ON h_link.child = child_ident.oidref "
            "JOIN basic AS parent_basic ON parent_basic.oid = h_link.parent "
            f"WHERE child_ident.id IN ({name_list})"
        )
        try:
            chunk_result = Simbad.query_tap(query)
        except Exception as e:
            print(f"  (parent lookup failed for a chunk of {len(chunk)} names: {e})")
            continue

        if chunk_result is None:
            continue

        for row in chunk_result:
            child_name = str(row["child_name"])
            if child_name in results and results[child_name] is None:
                parent_otype = row["parent_otype"]
                certainty = row["membership_certainty"]
                results[child_name] = (
                    strip_name_prefix(row["parent_name"]),
                    str(parent_otype) if parent_otype else "",
                    float(certainty) if certainty is not None and not is_value_masked(certainty) else None,
                )

    return results

def sanity_check_star(gaia_ids, teff_tolerance=2000, release="dr3"):
    """Cross-check Gaia's own data against SIMBAD's spectral type, to flag possible mismatches.

    Compares Gaia's photometric temperature (teff_gspphot) against the temperature range implied
    by SIMBAD's spectral type (e.g. "G2V"). A large difference can be a sign of a wrong cross-match 
    or another data issue worth double-checking.

    Args:
        gaia_ids (int or list[int]): Gaia source ID(s) to check.
        teff_tolerance (float, optional): How many Kelvin Gaia's teff_gspphot can fall outside the
            spectral-type-implied range before being flagged. Defaults to 2000 K.
        release (str, optional): Gaia data release to check against, "dr3" or "dr4". Note DR4's
            astrophysical parameters table layout is still in draft (DR4 is unreleased), so the
            table used for release="dr4" (ap_xp) is a best-effort guess and may need updating once
            DR4's final data model is published. Defaults to "dr3".

    Returns:
        pandas.DataFrame: One row per ID, with a 'common_name' column plus Gaia and SIMBAD fields
        side by side, and a 'flag' column (empty string if no mismatch was detected).
    """
    if isinstance(gaia_ids, int):
        gaia_ids = [gaia_ids]

    columns_out = ["gaia_id", "common_name", "gaia_g_mag", "gaia_teff", "simbad_sp_type", "simbad_v_mag", "flag"]
    if not gaia_ids:
        return pd.DataFrame(columns=columns_out)

    release_tables = GAIA_RELEASE_TABLES[release]

    # photometry + photometric temperature
    gaia_data = {}
    try:
        id_list = ",".join(str(g) for g in gaia_ids)
        gaia_query = f"""
        SELECT g.source_id, g.phot_g_mean_mag, ap.teff_gspphot
        FROM {release_tables["source_table"]} AS g
        JOIN {release_tables["ap_table"]} AS ap ON g.source_id = ap.source_id
        WHERE g.source_id IN ({id_list})
        """
        job = Gaia.launch_job(gaia_query)
        gaia_df = job.get_results().to_pandas()
        for _, row in gaia_df.iterrows():
            gaia_data[int(row["source_id"])] = (row.get("phot_g_mean_mag"), row.get("teff_gspphot"))
    except Exception as e:
        print(f"  (Gaia lookup for sanity check failed: {e})")

    # common name, spectral type, and V magnitude
    common_names, sp_types, v_mags = {}, {}, {}
    try:
        Simbad.add_votable_fields('sp_type', 'flux(V)')
    except Exception as e:
        print(f"  (could not request SIMBAD sp_type/V-mag fields: {e})")

    result = query_simbad_by_gaia_ids(gaia_ids, release, error_context="SIMBAD lookup for sanity check")

    if result is not None:
        for i, row in enumerate(result):
            gid = gaia_ids[i] if i < len(gaia_ids) else None
            if gid is None:
                continue
            try:
                if "main_id" in result.colnames and row["main_id"]:
                    common_names[gid] = strip_name_prefix(row["main_id"])
                if "sp_type" in result.colnames and row["sp_type"]:
                    sp_types[gid] = str(row["sp_type"])
                for v_col in ("flux_V", "FLUX_V"):
                    if v_col in result.colnames:
                        v_val = row[v_col]
                        if v_val is not None and not is_value_masked(v_val):
                            v_mags[gid] = float(v_val)
                            break
            except Exception:
                continue

    rows = []
    n_flagged = 0
    for gid in gaia_ids:
        gaia_g_mag, gaia_teff = gaia_data.get(gid, (None, None))
        sp_type = sp_types.get(gid)

        flag = ""
        teff_range = SPTYPE_TEFF_RANGES.get(str(sp_type).strip()[0:1].upper()) if sp_type else None
        if teff_range is not None and gaia_teff is not None and not pd.isna(gaia_teff):
            low, high = teff_range
            if gaia_teff < low - teff_tolerance or gaia_teff > high + teff_tolerance:
                flag = f"Possible mismatch: SIMBAD type '{sp_type}' implies ~{low}-{high} K, Gaia teff_gspphot={gaia_teff:.0f} K"
                n_flagged += 1

        rows.append({
            "gaia_id": gid,
            "common_name": common_names.get(gid),
            "gaia_g_mag": gaia_g_mag,
            "gaia_teff": gaia_teff,
            "simbad_sp_type": sp_type,
            "simbad_v_mag": v_mags.get(gid),
            "flag": flag,
        })

    df = pd.DataFrame(rows, columns=columns_out)
    print(f"Sanity check: {len(df)} source(s) checked, {n_flagged} flagged as possible mismatch(es).")
    if n_flagged > 0:
        for _, r in df[df["flag"] != ""].iterrows():
            print(f"  [{r['gaia_id']}] {r['flag']}")

    return df

def resolve_id(identifier, min_membership_certainty=80, center_ra=None, center_dec=None, radius_deg=None,
               mag_min=None, mag_max=None, mag_band="V", dist_min=None, dist_max=None,
               release="dr3", plot=True, save_plot=False, plot_file_name=None, label_points=None,
               sanity_check=False, save_csv=False, csv_file_name=None):
    """Convert a star or cluster identifier to its Gaia source ID(s), for the given data release.

    A numeric identifier is used as-is (assumed to be a valid Gaia source ID). Anything else is
    looked up via SIMBAD: a direct cross-match returns one Gaia ID; if there's no direct match,
    SIMBAD's hierarchy is checked for children of the identifier (e.g. cluster members), and every
    child that resolves to a Gaia ID is returned as a list.

    Cluster children go through several cleanup steps, each optional and printed as it happens:
    - min_membership_certainty filters by SIMBAD's own confidence score (0-100) for each member
      link, since a cluster's hierarchy often includes many low-confidence candidate members.
    - Children that are themselves groups (sub-clusters, associations) are skipped automatically.
    - Duplicate Gaia IDs (the same star reached via two different SIMBAD names) are removed.
    - center_ra/center_dec/radius_deg, mag_min/mag_max, and dist_min/dist_max are optional soft
      filters: a child is dropped only if it has that data on file AND it's outside the given range,
      missing data never excludes a child.

    The final result (including any unresolved children, with a blank Gaia ID) is shown as one
    table, with unresolved names also listed separately for easy scanning.

    Args:
        identifier (int or str): Gaia source ID, or any SIMBAD recognized star or cluster name/identifier.
        min_membership_certainty (int, optional): Minimum SIMBAD membership_certainty (0-100) a
            cluster child must have to be included. Set to None to disable this filter. Defaults to 80.
        center_ra (str or float, optional): RA to filter cluster children around, as a degree float
            or a sexagesimal string (e.g. "10h21m00s") when paired with a string center_dec. Only
            applied if center_dec and radius_deg are also given. Defaults to None (no radius filter).
        center_dec (str or float, optional): Dec to filter cluster children around, as a degree
            float or a sexagesimal string (e.g. "+41d05m00s") when paired with a string center_ra.
            Defaults to None.
        radius_deg (float, optional): Only keep cluster children within this many degrees of
            (center_ra, center_dec). Defaults to None (no radius filter).
        mag_min (float, optional): Faintest SIMBAD magnitude to include (soft filter). Defaults to
            None (no lower cutoff).
        mag_max (float, optional): Brightest SIMBAD magnitude to include (soft filter). Defaults to
            None (no upper cutoff).
        mag_band (str, optional): SIMBAD photometric band to filter on. Defaults to "V".
        dist_min (float, optional): Nearest distance to include, in parsecs, based on SIMBAD
            parallax (soft filter). Defaults to None (no lower cutoff).
        dist_max (float, optional): Farthest distance to include, in parsecs (soft filter).
            Defaults to None (no upper cutoff).
        release (str, optional): Gaia data release to resolve against, "dr3" or "dr4". Note DR4 is
            unreleased and SIMBAD has no Gaia DR4 cross-matches yet, so with release="dr4" a
            non-numeric identifier (a star/cluster name) will currently fail to resolve. Defaults to "dr3".
        plot (bool, optional): Whether to plot the resolved position(s). Defaults to True.
        save_plot (bool, optional): Whether to save the plot to an image file. Defaults to False.
        plot_file_name (str, optional): Name or path for the saved plot image. Defaults to None.
        label_points (bool, optional): Whether to label each plotted point with its name. Defaults
            to None (auto: labels are hidden if there are a lot of points, since that makes the
            plot unreadable - see plot_positions). Only relevant when identifier resolves to
            multiple children, since a single resolved star is always labeled.
        sanity_check (bool, optional): Whether to cross-check the resolved star(s) against SIMBAD's
            spectral type via sanity_check_star, and print any flagged mismatches. Adds 2 extra
            network calls when True. Defaults to False.
        save_csv (bool, optional): Whether to save the result to a CSV file, with columns
            'name', 'common_name', 'gaia_id', and 'binary_designations' (plus 'ra', 'dec', and
            'separation_deg' if plot or a radius filter was used). Matches the printed table
            exactly. Defaults to False.
        csv_file_name (str, optional): Name for the saved CSV file. Defaults to a name based on the
            identifier.

    Returns:
        int or list[int] or None: A single Gaia DR3 source ID if the identifier resolves directly, 
        a list of Gaia DR3 source IDs if it resolves to multiple children, or None if nothing
        could be resolved.
    """
    gaia_id = None
    common_name, binaries = None, []
    designation_prefix = GAIA_RELEASE_TABLES[release]["designation_prefix"]

    if str(identifier).strip().isdigit():
        gaia_id = int(identifier)
        identifier_table = Simbad.query_objectids(gaia_designation(gaia_id, release))
        if identifier_table is not None:
            common_name, binaries = find_name_and_binaries(identifier_table)
        if common_name:
            print(f"Resolved: {gaia_id} (numeric ID, used as-is) -> common name: {common_name}")
        else:
            print(f"Resolved: {gaia_id} (numeric ID, used as-is)")

    else:
        result = Simbad.query_objectids(identifier)
        if result is not None:
            common_name, binaries = find_name_and_binaries(result)
            for row in result["id"]:
                if row.startswith(designation_prefix + " "):
                    gaia_id = int(row.replace(designation_prefix + " ", ""))
                    break

        if gaia_id is not None:
            if common_name and common_name != identifier:
                print(f"Resolved: '{identifier}' ({common_name}) -> {designation_prefix} {gaia_id}")
            else:
                print(f"Resolved: '{identifier}' -> {designation_prefix} {gaia_id}")
        elif release == "dr4":
            print(
                f"Could not resolve '{identifier}'. Note: SIMBAD has no Gaia DR4 cross-matches yet "
                "(DR4 is unreleased) - only numeric Gaia DR4 source IDs can be resolved for now."
            )

    if gaia_id is not None:
        if binaries:
            print(f"  Binary/multiple star designation(s) found: {binaries}")

        if plot:
            points = fetch_coordinates(gaia_id, release=release)
            if common_name:
                points = [(common_name, p_ra, p_dec) for _, p_ra, p_dec in points]
            plot_positions(
                points, title=f"Cross-match position for '{identifier}'",
                save_plot=save_plot, plot_file_name=plot_file_name, default_file_name=f"resolve_{gaia_id}",
            )

        if sanity_check:
            sanity_check_star(gaia_id, release=release)

        if save_csv:
            csv_df = pd.DataFrame([{
                "name": str(identifier),
                "common_name": common_name,
                "gaia_id": gaia_id,
                "binary_designations": "; ".join(binaries) if binaries else None,
            }])
            save_dataframe_csv(csv_df, csv_file_name, f"resolve_{gaia_id}")

        return gaia_id

    # No direct cross-match
    hierarchy_criteria = (
        f"h_link.membership >= {min_membership_certainty}" if min_membership_certainty is not None else None
    )

    Simbad.add_votable_fields('otype')
    children = Simbad.query_hierarchy(identifier, hierarchy="children", criteria=hierarchy_criteria)
    if children is None or len(children) == 0:
        note = ""
        if release == "dr4":
            note = " Note: SIMBAD has no Gaia DR4 cross-matches yet (DR4 is unreleased)."
        print(f"Could not resolve '{identifier}'.{note}")
        return None

    raw_count = len(children)

    # Group-type objects
    group_otypes = ("Cl*", "OpC", "GlC", "As*", "SCG")
    seen_names = set()
    child_names = []
    skipped_groups = 0
    for row in children:
        name = strip_name_prefix(row["main_id"])
        if name in seen_names:
            continue
        seen_names.add(name)
        if "otype" in children.colnames and str(row["otype"]).startswith(group_otypes):
            skipped_groups += 1
            continue
        child_names.append(name)
    unique_name_count = len(seen_names)

    batch_results = resolve_children_batch(child_names, release=release)

    all_rows = []
    for child_name in child_names:
        info = batch_results.get(child_name, {})
        binaries = info.get("binaries", [])
        all_rows.append({
            "name": child_name,
            "common_name": info.get("common_name"),
            "gaia_id": info.get("gaia_id"),
            "binary_designations": "; ".join(binaries) if binaries else None,
        })
    df = pd.DataFrame(all_rows, columns=["name", "common_name", "gaia_id", "binary_designations"])

    # Deduplicate by Gaia ID only among resolved rows, unresolved rows are always kept
    resolved_mask = df["gaia_id"].notna()
    resolved_before_dedup = int(resolved_mask.sum())
    unresolved_count = int((~resolved_mask).sum())
    resolved_part = df[resolved_mask].drop_duplicates(subset="gaia_id", keep="first")
    resolved_part["gaia_id"] = resolved_part["gaia_id"].astype(int)
    df = pd.concat([resolved_part, df[~resolved_mask]], ignore_index=True)

    if df.empty:
        print(f"'{identifier}' has no {designation_prefix} cross-match, and none of its children do either.")
        return None

    certainty_note = f", membership_certainty >= {min_membership_certainty}" if min_membership_certainty is not None else ""
    print(f"Resolving '{identifier}'{certainty_note}:")
    print(f"  -> {raw_count} raw membership link(s) from SIMBAD.")
    print(f"  -> {unique_name_count} unique star name(s) after removing duplicates.")
    if skipped_groups:
        print(f"  -> {len(child_names)} individual star(s) after removing {skipped_groups} group-type entries (sub-clusters/associations).")
    print(f"  -> {resolved_before_dedup} resolved to a {designation_prefix} ID ({unresolved_count} could not be resolved).")
    if resolved_before_dedup != len(resolved_part):
        print(
            f"  -> {len(resolved_part)} resolved star(s) final, after removing "
            f"{resolved_before_dedup - len(resolved_part)} duplicate Gaia ID(s) (same star via a different SIMBAD name)."
        )

    want_radius_filter = center_ra is not None and center_dec is not None and radius_deg is not None
    if want_radius_filter:
        center_ra, center_dec = resolve_search_position(center_ra, center_dec)
    if plot or want_radius_filter:
        resolved_ids_for_coords = df.loc[df["gaia_id"].notna(), "gaia_id"].astype(int).tolist()
        points = fetch_coordinates(resolved_ids_for_coords, release=release)
        coord_map = {}
        if points and len(points) == len(resolved_ids_for_coords):
            for gid, (_, p_ra, p_dec) in zip(resolved_ids_for_coords, points):
                coord_map[gid] = (p_ra, p_dec)
        df["ra"] = df["gaia_id"].apply(lambda g: coord_map.get(int(g), (None, None))[0] if pd.notna(g) else None)
        df["dec"] = df["gaia_id"].apply(lambda g: coord_map.get(int(g), (None, None))[1] if pd.notna(g) else None)

    if want_radius_filter and "ra" in df.columns:
        center_coord = SkyCoord(ra=center_ra * u.deg, dec=center_dec * u.deg)
        df["separation_deg"] = df.apply(
            lambda r: center_coord.separation(SkyCoord(ra=r["ra"] * u.deg, dec=r["dec"] * u.deg)).deg
            if pd.notna(r["ra"]) and pd.notna(r["dec"]) else None,
            axis=1,
        )
        before_radius = len(df)

        df = df[df["separation_deg"].isna() | (df["separation_deg"] <= radius_deg)].reset_index(drop=True)
        print(f"  -> {len(df)} final, after dropping {before_radius - len(df)} outside {radius_deg} deg of (ra={center_ra}, dec={center_dec}).")

    want_mag_filter = mag_min is not None or mag_max is not None
    want_dist_filter = dist_min is not None or dist_max is not None
    if want_mag_filter or want_dist_filter:
        gaia_ids_for_lookup = df.loc[df["gaia_id"].notna(), "gaia_id"].astype(int).tolist()
        mag_plx_map = fetch_mag_and_parallax(gaia_ids_for_lookup, mag_band=mag_band, release=release)
        df["mag"] = df["gaia_id"].apply(lambda g: mag_plx_map.get(int(g), {}).get("mag") if pd.notna(g) else None)
        df["plx_value"] = df["gaia_id"].apply(lambda g: mag_plx_map.get(int(g), {}).get("plx_value") if pd.notna(g) else None)
        df["distance_pc"] = df["plx_value"].apply(lambda p: 1000 / p if pd.notna(p) and p > 0 else None)

    if want_mag_filter:
        before_mag = len(df)
        mag_in_range = (
            (mag_min is None or df["mag"] >= mag_min) & (mag_max is None or df["mag"] <= mag_max)
        )
        df = df[df["mag"].isna() | mag_in_range].reset_index(drop=True)
        print(f"  -> {len(df)} final, after dropping {before_mag - len(df)} outside {mag_band}-band magnitude range [{mag_min}, {mag_max}].")

    if want_dist_filter:
        before_dist = len(df)
        dist_in_range = (
            (dist_min is None or df["distance_pc"] >= dist_min) & (dist_max is None or df["distance_pc"] <= dist_max)
        )
        df = df[df["distance_pc"].isna() | dist_in_range].reset_index(drop=True)
        print(f"  -> {len(df)} final, after dropping {before_dist - len(df)} outside distance range [{dist_min}, {dist_max}] pc.")

    if save_csv:
        save_dataframe_csv(df, csv_file_name, f"{identifier}_children")

    with pd.option_context("display.width", None, "display.max_columns", None):
        print(df)

    unresolved_names = df.loc[df["gaia_id"].isna(), "name"].tolist()
    if unresolved_names:
        print(f"Could not resolve {len(unresolved_names)} child(ren) of '{identifier}': {unresolved_names}")

    if plot:
        plot_points = [
            (row["common_name"] if pd.notna(row["common_name"]) else row["name"], row["ra"], row["dec"])
            for _, row in df.iterrows() if "ra" in df.columns and pd.notna(row["ra"]) and pd.notna(row["dec"])
        ]
        plot_positions(plot_points, title=f"Cross-match positions for children of '{identifier}'", save_plot=save_plot, plot_file_name=plot_file_name, default_file_name=f"{identifier}_children", label_points=label_points)

    resolved_ids = df.loc[df["gaia_id"].notna(), "gaia_id"].astype(int).tolist()

    if sanity_check:
        sanity_check_star(resolved_ids, release=release)

    return resolved_ids

def query_by_adql(adql_query: str = None, identifier: int | str = None, release: str = "dr3", save_file: bool = False, file_name: str = None):
    """Query Gaia with an ADQL query.

    If 'identifier' is given, it is resolved to a Gaia source ID (via resolve_id) and 
    substituted into 'adql_query' wherever the placeholder '{source_id}' appears. If the 
    identifier resolves to multiple children, all resolved IDs are substituted in as a comma-separated list.

    If 'adql_query' is not given but 'identifier' is, a default query of 
    'SELECT * FROM <source_table> WHERE source_id IN ({source_id})' is used automatically (where
    <source_table> depends on 'release'), so an identifier is enough to look up a star (or every
    member of a cluster) without writing any ADQL.

    Note:
        Because an identifier can resolve to more than one Gaia ID, any custom 'adql_query' should 
        use an 'IN ({source_id})' clause rather than '= {source_id}'. With '=', a query will work for a 
        single resolved ID, but will fail with an ADQL syntax error if 'identifier' resolves to multiple IDs.

    Args:
        adql_query (str, optional): ADQL query targeting a Gaia table. May contain a '{source_id}' placeholder 
            to be filled in from 'identifier'. If omitted, 'identifier' must be given, and a default 'SELECT *' 
            query is built automatically. Defaults to None.
        identifier (int or str, optional): Gaia source ID or other star/cluster identifier to resolve and 
            insert into the query in place of '{source_id}'. Defaults to None.
        release (str, optional): Gaia data release to resolve 'identifier' against and to use for
            the auto-built default query, "dr3" or "dr4". Only used when 'identifier' is given.
            A custom 'adql_query' already names its own table(s). Defaults to "dr3".
        save_file (bool, optional): Whether to save the result to a CSV file. Defaults to False.
        file_name (str, optional): Name or path for the saved CSV file. Defaults to the resolved Gaia ID(s) 
            if an identifier was given, or to "gaia_query" otherwise.

    Returns:
        pandas.DataFrame: Query results.

    Raises:
        ValueError: If neither 'adql_query' nor 'identifier' is given.
    """
    if adql_query is None and identifier is None:
        raise ValueError("Provide either adql_query, identifier, or both.")

    if identifier is not None:
        gaia_id = resolve_id(identifier, release=release)

        if gaia_id is None:
            print(f"Could not resolve '{identifier}' to a Gaia source ID; query not run.")
            return None

        if isinstance(gaia_id, list):
            source_id_str = ",".join(str(i) for i in gaia_id)
            default_name = "_".join(str(i) for i in gaia_id)
        else:
            source_id_str = str(gaia_id)
            default_name = str(gaia_id)

        if file_name is None:
            file_name = default_name

        if adql_query is None:
            source_table = GAIA_RELEASE_TABLES[release]["source_table"]
            adql_query = f"SELECT * FROM {source_table} WHERE source_id IN " + "({source_id})"
        adql_query = adql_query.format(source_id=source_id_str)

    job = Gaia.launch_job(adql_query)
    df = job.get_results().to_pandas()

    if save_file:
        if file_name is None:
            file_name = "gaia_query"
        formatted_name = file_name.replace(" ", "_")
        formatted_name = f"{formatted_name}.csv"
        df.to_csv(formatted_name)

    return df

def find_star(
    ra: str | float = None,
    dec: str | float = None,
    coordinates: SkyCoord = None,
    degree_range: float = 0.0001,
    columns: str = '*',
    mag_min: float = None,
    mag_max: float = None,
    mag_band: str = "phot_g_mean_mag",
    dist_min: float = None,
    dist_max: float = None,
    release: str = "dr3",
    plot: bool = True,
    save_plot: bool = False,
    plot_file_name: str = None,
    label_points: bool = None,
    save_csv: bool = False,
    csv_file_name: str = "star_query"
):
    """Query Gaia for sources within a small radius of a sky position.

    The position can be supplied either as sexagesimal RA/Dec strings, or as an astropy SkyCoord. 

    Magnitude filtering is a hard filter, applied directly in the ADQL query. 
    Distance filtering is a soft filter, applied after the query, a source is only
    excluded if it has a parallax on file and its implied distance is outside the given range.

    Args:
        ra (str or float, optional): Right ascension, as a sexagesimal string (e.g. "10h21m00s") when paired 
            with a string `dec`. Defaults to None.
        dec (str or float, optional): Declination, as a sexagesimal string (e.g. "+41d05m00s") when paired with 
            a string `ra`. Defaults to None.
        coordinates (astropy.coordinates.SkyCoord, optional): Sky position to search around, used if `ra`/`dec` are 
            not given. Defaults to None.
        degree_range (float, optional): Search radius in degrees. Defaults to 0.0001.
        columns (str, optional): Comma-separated columns to select from the release's source table. Defaults to '*' (all columns).
        mag_min (float, optional): Faintest magnitude to include (hard filter). Defaults to None (no lower cutoff).
        mag_max (float, optional): Brightest magnitude to include (hard filter). Defaults to None (no upper cutoff).
        mag_band (str, optional): Gaia magnitude column to filter on. Defaults to "phot_g_mean_mag".
        dist_min (float, optional): Nearest distance to include, in parsecs (soft filter). Defaults to
            None (no lower cutoff).
        dist_max (float, optional): Farthest distance to include, in parsecs (soft filter). Defaults to
            None (no upper cutoff).
        release (str, optional): Gaia data release to query, "dr3" or "dr4". Note SIMBAD has no
            Gaia DR4 cross-matches yet (DR4 is unreleased), so with release="dr4" the returned
            'common_name' column will currently be blank for every source. Defaults to "dr3".
        plot (bool, optional): Whether to plot the found source(s). Defaults to True.
        save_plot (bool, optional): Whether to save the plot to an image file. Defaults to False.
        plot_file_name (str, optional): Name or path for the saved plot image. Defaults to None.
        label_points (bool, optional): Whether to label each plotted source with its name.
            Defaults to None (auto: labels are hidden if there are a lot of sources, since that
            makes the plot unreadable - see plot_positions).
        save_csv (bool, optional): Whether to save the result to a CSV file. Defaults to False.
        csv_file_name (str, optional): Name for the saved CSV file. Defaults to "star_query".

    Returns:
        pandas.DataFrame: Query results, with an added 'common_name' column from SIMBAD - a true
        proper name (e.g. "Proxima Centauri") if one is on file, else SIMBAD's main identifier,
        else blank if SIMBAD has no record of the source at all.

    Raises:
        ValueError: If neither a valid ra/dec pair nor coordinates is given.
    """
    ra, dec = resolve_search_position(ra, dec, coordinates)

    conditions = []
    if mag_min is not None:
        conditions.append(f"{mag_band} >= {mag_min}")
    if mag_max is not None:
        conditions.append(f"{mag_band} <= {mag_max}")

    extra_filter = ""
    if conditions:
        extra_filter = " AND " + " AND ".join(conditions)

    want_dist_filter = dist_min is not None or dist_max is not None
    query_columns = columns
    if want_dist_filter and columns != '*':
        existing = [c.strip() for c in columns.split(",")]
        if "parallax" not in existing:
            query_columns = columns + ",parallax"

    source_table = GAIA_RELEASE_TABLES[release]["source_table"]
    query = f"""
    SELECT TOP 10 {query_columns}
    FROM {source_table}
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {degree_range})
    ) = 1{extra_filter}
    """
    if save_csv:
        df = query_by_adql(query, save_file=True, file_name=csv_file_name)
    else:
        df = query_by_adql(query)

    if df is not None and "source_id" in df.columns:
        common_names = fetch_common_names(df["source_id"].tolist(), release=release)
        df["common_name"] = df["source_id"].map(common_names)

        cols = df.columns.tolist()
        cols.remove("common_name")
        source_id_idx = cols.index("source_id")
        cols.insert(source_id_idx + 1, "common_name")
        df = df[cols]

    dropped_by_distance = 0
    if df is not None and want_dist_filter and "parallax" in df.columns:
        before_dist = len(df)
        df["implied_distance_pc"] = df["parallax"].apply(lambda p: 1000 / p if pd.notna(p) and p > 0 else None)
        dist_in_range = (
            (dist_min is None or df["implied_distance_pc"] >= dist_min)
            & (dist_max is None or df["implied_distance_pc"] <= dist_max)
        )
        df = df[df["implied_distance_pc"].isna() | dist_in_range].reset_index(drop=True)
        dropped_by_distance = before_dist - len(df)

    if df is not None:
        print(f"Cone search near (ra={ra:.4f}, dec={dec:.4f}), radius {degree_range} deg:")
        if conditions:
            print(f"  -> filter applied: {' AND '.join(conditions)}")
        print(f"  -> {len(df) + dropped_by_distance} source(s) found.")
        if release == "dr4":
            print("  -> Note: common names unavailable for DR4 sources until SIMBAD ingests DR4 cross-matches.")
        if want_dist_filter:
            print(f"  -> {len(df)} final, after dropping {dropped_by_distance} outside distance range [{dist_min}, {dist_max}] pc.")
        if "non_single_star" in df.columns and "source_id" in df.columns:
            flagged = df[df["non_single_star"] > 0]
            if len(flagged) > 0:
                print(f"  -> {len(flagged)} source(s) flagged as non-single-star (binary/multiple) system(s):")
                for _, row in flagged.iterrows():
                    flags = int(row["non_single_star"])
                    types = []
                    if flags & 1:
                        types.append("astrometric")
                    if flags & 2:
                        types.append("spectroscopic")
                    if flags & 4:
                        types.append("eclipsing")
                    print(f"    source_id {row['source_id']}: {', '.join(types) if types else f'flag={flags}'}")

    if plot and df is not None and "ra" in df.columns and "dec" in df.columns:
        points = [
            (row["common_name"] if "common_name" in df.columns and pd.notna(row["common_name"]) else str(row.get("source_id", i)), row["ra"], row["dec"])
            for i, row in df.iterrows()
        ]
        plot_positions(
            points,
            title=f"Cone search near (ra={ra:.4f}, dec={dec:.4f})",
            save_plot=save_plot,
            plot_file_name=plot_file_name,
            center=(ra, dec),
            radius_deg=degree_range,
            default_file_name=f"star_search_ra{ra:.2f}_dec{dec:.2f}",
            label_points=label_points,
        )

    return df

def find_cluster(
    ra: str | float = None,
    dec: str | float = None,
    coordinates: SkyCoord = None,
    search_radius_deg: float = 2.0,
    top_n: int = 10,
    min_membership_certainty: int = 80,
    dist_min: float = None,
    dist_max: float = None,
    plot: bool = True,
    save_plot: bool = False,
    plot_file_name: str = None,
    label_points: bool = None,
    save_csv: bool = False,
    csv_file_name: str = "cluster_query"
):
    """Find clusters near a sky position, using SIMBAD,identified by their members.

    Distance filtering is a soft filter

    min_membership_certainty, if set, only counts a member if its SIMBAD confidence score meets
    that bar; a member with no score does NOT count. Off by default.

    Args:
        ra (str or float, optional): Right ascension, as a sexagesimal string (e.g. "10h21m00s") when paired
            with a string `dec`. Defaults to None.
        dec (str or float, optional): Declination, as a sexagesimal string (e.g. "+41d05m00s") when paired
            with a string `ra`. Defaults to None.
        coordinates (astropy.coordinates.SkyCoord, optional): Sky position to search around, used if `ra`/`dec`
            are not given. Defaults to None.
        search_radius_deg (float, optional): Search radius in degrees. Defaults to 2.0.
        top_n (int, optional): Maximum number of candidates to return, ranked by member_count.
            Defaults to 10.
        min_membership_certainty (int, optional): Minimum SIMBAD membership_certainty (0-100) a
            member must have to count toward its cluster. A member with no certainty score is
            excluded when this is set. Set to None to disable this filter. Defaults to 80.
        dist_min (float, optional): Nearest implied distance to include, in parsecs (soft filter).
            Defaults to None (no lower cutoff).
        dist_max (float, optional): Farthest implied distance to include, in parsecs (soft filter).
            Defaults to None (no upper cutoff).
        plot (bool, optional): Whether to plot the returned candidate(s). Defaults to True.
        save_plot (bool, optional): Whether to save the plot to an image file. Defaults to False.
        plot_file_name (str, optional): Name or path for the saved plot image. Defaults to None.
        label_points (bool, optional): Whether to label each plotted candidate with its name.
            Defaults to None (auto: labels are hidden if there are a lot of candidates, since that
            makes the plot unreadable - see plot_positions).
        save_csv (bool, optional): Whether to save the result to a CSV file. Defaults to False.
        csv_file_name (str, optional): Name for the saved CSV file. Defaults to "cluster_query".

    Returns:
        pandas.DataFrame: Up to 'top_n' candidate clusters, sorted by member_count (descending),
        with columns 'name', 'common_name' (a true proper name if SIMBAD has one on file, else
        blank), 'member_count', 'ra', 'dec' (centroid of found members), 'otype', 'separation_deg'
        (of the centroid from the search position), and 'implied_distance_pc'. Empty (but same
        columns) if nothing is found.

    Raises:
        ValueError: If neither a valid ra/dec pair nor coordinates is given.
    """
    ra, dec = resolve_search_position(ra, dec, coordinates)

    columns_out = [
        "name", "common_name", "member_count", "ra", "dec", "otype",
        "separation_deg", "implied_distance_pc",
    ]

    print(f"Cluster search near (ra={ra}, dec={dec}), radius {search_radius_deg} deg:")

    Simbad.add_votable_fields('parallax')

    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    result = Simbad.query_region(coord, radius=search_radius_deg * u.deg)

    if result is None:
        print("  -> no SIMBAD objects found.")
        return pd.DataFrame(columns=columns_out)

    member_names = []
    member_info = {}
    for row in result:
        name = str(row["main_id"])
        try:
            if "ra" in result.colnames and "dec" in result.colnames:
                row_ra, row_dec = row["ra"], row["dec"]
            else:
                row_ra, row_dec = row["RA"], row["DEC"]
            if isinstance(row_ra, str):
                row_coord = SkyCoord(ra=row_ra, dec=row_dec, unit=(u.hourangle, u.deg))
            else:
                row_coord = SkyCoord(ra=row_ra * u.deg, dec=row_dec * u.deg)
        except Exception:
            continue

        plx = row["plx_value"] if "plx_value" in result.colnames else None
        plx_val = float(plx) if plx and plx > 0 else None

        member_names.append(name)
        member_info[name] = {"ra": row_coord.ra.deg, "dec": row_coord.dec.deg, "plx_value": plx_val}

    if not member_names:
        print("  -> no usable star positions found.")
        return pd.DataFrame(columns=columns_out)

    print(f"  -> {len(member_names)} star(s) found.")

    parent_map = resolve_parents_batch(member_names)
    groups = {}
    skipped_uncertain = 0
    for name in member_names:
        parent = parent_map.get(name)
        if parent is None:
            continue
        parent_name, parent_otype, certainty = parent

        if min_membership_certainty is not None:
            if certainty is None or certainty < min_membership_certainty:
                skipped_uncertain += 1
                continue

        groups.setdefault(parent_name, {"otype": parent_otype, "members": []})["members"].append(name)

    if min_membership_certainty is not None and skipped_uncertain:
        print(f"  -> skipped {skipped_uncertain} member link(s) below membership_certainty {min_membership_certainty}.")

    if not groups:
        print("  -> none had a resolvable parent cluster.")
        return pd.DataFrame(columns=columns_out)

    print(f"  -> grouped into {len(groups)} candidate cluster(s).")

    common_name_map = resolve_children_batch(list(groups.keys()))

    candidates = []
    for parent_name, info in groups.items():
        members = info["members"]
        centroid_ra = sum(member_info[m]["ra"] for m in members) / len(members)
        centroid_dec = sum(member_info[m]["dec"] for m in members) / len(members)
        sep = coord.separation(SkyCoord(ra=centroid_ra * u.deg, dec=centroid_dec * u.deg)).deg

        plx_values = [member_info[m]["plx_value"] for m in members if member_info[m]["plx_value"] is not None]
        implied_distance = 1000 / statistics.median(plx_values) if plx_values else None

        if implied_distance is not None:
            if dist_min is not None and implied_distance < dist_min:
                continue
            if dist_max is not None and implied_distance > dist_max:
                continue

        candidates.append({
            "name": parent_name,
            "common_name": common_name_map.get(parent_name, {}).get("common_name"),
            "member_count": len(members),
            "ra": centroid_ra,
            "dec": centroid_dec,
            "otype": info["otype"],
            "separation_deg": sep,
            "implied_distance_pc": implied_distance,
        })

    if not candidates:
        print("  -> none passed the distance filter.")
        return pd.DataFrame(columns=columns_out)

    df = pd.DataFrame(candidates).sort_values(
        ["member_count", "separation_deg"], ascending=[False, True]
    ).head(top_n).reset_index(drop=True)

    print(f"  -> returning top {len(df)} by member count.")

    if save_csv:
        save_dataframe_csv(df, csv_file_name, "cluster_query")

    if plot and "ra" in df.columns and "dec" in df.columns:
        points = [
            (row["common_name"] if pd.notna(row["common_name"]) else row["name"], row["ra"], row["dec"])
            for _, row in df.iterrows()
        ]
        plot_positions(
            points,
            title=f"Candidate clusters near (ra={ra}, dec={dec})",
            save_plot=save_plot,
            plot_file_name=plot_file_name,
            center=(ra, dec),
            radius_deg=search_radius_deg,
            default_file_name=f"cluster_search_ra{ra:.2f}_dec{dec:.2f}",
            label_points=label_points,
        )

    return df

def load_csv(file_path):
    """Load a Gaia CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)

def query_by_datalink(
    gaia_ids: int | list[int],
    retrieval: str = 'EPOCH_PHOTOMETRY',
    structure: str = 'INDIVIDUAL',
    release: str = 'dr3',
    save_file: bool = False,
    folder_name: str = None
):
    """Query Gaia with a Datalink query.

    Any non-Gaia identifiers are automatically resolved to Gaia source IDs via SIMBAD (see resolve_id) before 
    the query is run. Identifiers that resolve to multiple children are expanded so every resolved star is included.

    Args:
        gaia_ids (int or str or list[int or str]): Gaia source ID(s) or other star/cluster identifier(s) 
            (resolved automatically) of the target(s) of the query.
        retrieval (str, optional): Retrieval type. Defaults to 'EPOCH_PHOTOMETRY'.
        structure (str, optional): Data structure. Defaults to 'INDIVIDUAL'.
        release (str, optional): Gaia data release, "dr3" or "dr4". Note DR4 is unreleased and
            SIMBAD has no Gaia DR4 cross-matches yet, so with release="dr4" any non-numeric
            identifier (a star/cluster name) will currently fail to resolve - only numeric Gaia
            DR4 source IDs can be resolved for now. Defaults to "dr3".
        save_file (bool, optional): Whether to save each result to a CSV file. Defaults to False.
        folder_name (str, optional): Folder to save CSV files into. Required if save_file is True. Defaults to None.

    Returns:
        dict: Query results. Keys are Gaia IDs, values are epoch photometry DataFrames.

    Raises:
        TypeError: If save_file is True and folder_name is not a string.
    """
    if save_file and not isinstance(folder_name, str):
        raise TypeError(f"Expected string data for folder_name, got {type(folder_name)}")

    if isinstance(gaia_ids, (int, str)):
        gaia_ids = [gaia_ids]

    if release == "dr4" and any(not str(gid).strip().isdigit() for gid in gaia_ids):
        print(
            "Note: SIMBAD has no Gaia DR4 cross-matches yet (DR4 is unreleased) - "
            "only numeric Gaia DR4 source IDs can be resolved for now."
        )

    # resolve_id may return a single int (direct match) or a list[int]
    resolved_ids = []
    for gid in gaia_ids:
        resolved = resolve_id(gid, release=release)
        if resolved is None:
            continue
        if isinstance(resolved, list):
            resolved_ids.extend(resolved)
        else:
            resolved_ids.append(resolved)

    if not resolved_ids:
        print("No IDs could be resolved.")
        return {}

    datalink_release = GAIA_RELEASE_TABLES[release]["datalink_release"]
    dl_query = Gaia.load_data(ids=resolved_ids, data_release=datalink_release, retrieval_type=retrieval, data_structure=structure)
    df_dict = {}
    retrieved_ids = set()

    for key, value in dl_query.items():
        gaia_id = int(key[26:-4])
        retrieved_ids.add(gaia_id)
        df = value[0].to_table().to_pandas()
        df_dict[gaia_id] = df

        if save_file:
            file_name = folder_name + "/" + str(gaia_id) + ".csv"
            df.to_csv(file_name)

    if len(retrieved_ids) <= 0:
        print("No IDs could be retrieved. (no Epoch Photometry data).")
    else:
        not_retrieved = set(resolved_ids) - retrieved_ids
        if not_retrieved:
            print(f"IDs not retrieved (no Epoch Photometry data):\n{not_retrieved}")

    return df_dict

def apply_filter(df):
    """Filter a dataframe, or each dataframe in a dict of dataframes, using a pandas query expression.

    If 'df' is a dict of DataFrames (query_by_datalink), the same filter expression is applied to every DataFrame 
    in the dict, and a dict of filtered results is returned. The column list shown to the user is taken from the 
    first entry in the dict, since epoch photometry tables share the same schema.

    Args:
        df (pandas.DataFrame or dict[int, pandas.DataFrame]): Dataframe, or dict of dataframes, to filter.

    Returns:
        pandas.DataFrame or dict[int, pandas.DataFrame]: Filtered dataframe(s), or original if skipped/invalid.
    """
    if isinstance(df, dict):
        if not df:
            return df

        sample = next(iter(df.values()))
        print("Columns:", sample.columns.tolist())
        expression = input("Filter expression (e.g. parallax > 1.0), or Enter to skip: ").strip()

        if not expression:
            return df

        filtered = {}
        for key, value in df.items():
            try:
                filtered[key] = value.query(expression)
            except Exception as e:
                print(f"Invalid filter for ID {key}: {e}")
                filtered[key] = value
        return filtered

    print("Columns:", df.columns.tolist())
    expression = input("Filter expression (e.g. parallax > 1.0), or Enter to skip: ").strip()

    if not expression:
        return df
    try:
        return df.query(expression)
    except Exception as e:
        print(f"Invalid filter: {e}")
        return df

def gaia_login_prompt():
    """Prompt the user to optionally log in to the Gaia archive."""
    if input("Log in to Gaia archive? (y/n): ").strip().lower() == "y":
        user = input("Gaia username: ").strip()
        password = input("Gaia password: ").strip()
        try:
            Gaia.login(user=user, password=password)
            print("Logged in to Gaia archive.")
        except Exception as e:
            print(f"Login failed: {e}")

def get_dataframe():
    """Run the input menu and return a Gaia dataframe.

    Returns:
        pandas.DataFrame: Final dataframe for downstream use.
    """
    gaia_login_prompt()

    print("\n1. ADQL query\n2. CSV file\n3. Datalink (Epoch Photometry)")
    choice = input("Choose input method (1/2/3): ").strip()

    if choice == "1":
        use_identifier = input(
            "Search by identifier (Gaia ID, star name, cluster, etc.) instead of writing a full ADQL query? (y/n): "
        ).strip().lower() == "y"

        if use_identifier:
            identifier = input("Identifier: ").strip()
            adql_query = None
        else:
            adql_query = input("ADQL query (use {source_id} as a placeholder to resolve an identifier into it): ").strip()
            id_input = input("Identifier to resolve into {source_id}, or Enter to skip: ").strip()
            identifier = id_input or None

        if input("Save to csv file? (y/n): ").strip().lower() == "y":
            path = input("Enter name for file OR path to file (press Enter for default name): ").strip()
            if path != "":
                df = query_by_adql(adql_query, save_file=True, file_name=path, identifier=identifier)
            else:
                df = query_by_adql(adql_query, save_file=True, identifier=identifier)
        else:
            df = query_by_adql(adql_query, identifier=identifier)

    elif choice == "2":
        filepath = input("CSV path: ").strip()
        df = load_csv(filepath)

    elif choice == "3":
        raw = input("Gaia source ID(s) or star name(s), comma-separated: ").strip()
        gaia_ids = [i.strip() for i in raw.split(",")]

        if input("Save to csv file? (y/n): ").strip().lower() == "y":
            path = input("Enter folder name OR path to folder (folder must exist): ").strip()
            if path != "":
                df = query_by_datalink(gaia_ids, save_file=True, folder_name=path)
            else:
                df = query_by_datalink(gaia_ids, save_file=True, folder_name=".")
        else:
            df = query_by_datalink(gaia_ids)
    else:
        print("Invalid choice.")
        return None

    return df

if __name__ == "__main__":
    df = get_dataframe()
    if df is not None:
        print(df.head())