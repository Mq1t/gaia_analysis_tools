import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator
import os

from .constants import MAIN_SEQUENCE_TABLE, HRD_REGION_LABELS

default_folder = None 
style = 'seaborn-v0_8-darkgrid'
plt.style.use(style)

#Create a Ra vs Dec diagram.
def ra_vs_dec(
        df: pd.DataFrame, 
        error: bool = True,
        xlim: int|float = None, 
        ylim: int|float = None, 
        color: str ='red', 
        size: int|float = 0.7, 
        title: str = 'Right Ascension Vs. Declination', 
        save_plot: bool = False, 
        file_name: str | None = 'ra_vs_dec', 
        save_folder: str = default_folder):
    """
    Plot Right Ascension (RA) vs Declination (Dec) from a pandas DataFrame.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing at least two columns, 'ra' for Right Ascension
                          and 'dec' for Declination.
        error (bool): If true the plot will include error bars using required columns 'ra_error', and 'dec_error'.
        xlim (int|float, optional): The x-axis upper limit. If None, the default limits are used. Default is None.
        ylim (int|float, optional): The y-axis upper limit. If None, the default limits are used. Default is None.
        color (str, optional): Color of the plotted points. Default is 'red'.
        size (int|float, optional): Size of the plotted points. Default is 0.5.
        title (str, optional): Title of the plot. Default is 'Right Ascension Vs. Declination'.
        save_plot (bool, optional): If true, saves plot as a PDF file. Defaults to False. 
        file_name (str, optional): File name of the resulting plot. Default is 'ra_vs_dec'. File identifier is added automatically.
        save_folder (str, optional): Optional folder destination. A destination folder could also be set using the file name.
    
    Returns:
        None

    Raises:
        TypeError: If the input data is not a pandas DataFrame.
        KeyError: If the required columns are missing ('ra', 'dec')
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Data must be of type pandas.DataFrame')

    required_cols = set()
    
    # Ensure required columns exist
    if error == False:
        required_cols.update({'ra', 'dec'})

    else:
        required_cols.update({'ra', 'dec', 'ra_error', 'dec_error'})
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame is missing required columns: {', '.join(sorted(missing))}")

    # RA, X-Value
    x = df['ra']
    # Declination, Y-Values
    y = df['dec']


    if error:
        #Converts error from mas to degrees
        xerr = df['ra_error'] / 3600000
        yerr = df['dec_error'] / 3600000
        plt.errorbar(
            x, y,
            xerr=xerr,
            yerr=yerr,
            fmt='none',
            markersize=size,
            ecolor=color,       # error bar color
            elinewidth=0.5,
            capsize=2,
            alpha=0.4,           # reduce clutter, makes it slightly transparetn
            zorder=1
        )
    plt.scatter(x, y, c=color, s=size, zorder=3)
    plt.title(title)
    plt.xlabel("RA")
    plt.ylabel("Dec")

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if save_plot:
        safe_name = file_name.replace(" ", "_")
        safe_name = f"{safe_name}.pdf"
        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)  
            filepath = os.path.join(save_folder, safe_name)
        else:
            filepath = safe_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filepath}")
    plt.show()

#Proper motion
def pmra_vs_pmdec(
        df: pd.DataFrame, 
        error:bool = True,
        xlim:float = None, 
        ylim:float = None, 
        color: str ='red', 
        size: int|float = 0.5, 
        title: str = 'Right Ascension Vs. Declination in Proper Motion Space', 
        save_plot: bool = False, 
        file_name: str | None = 'pmra_vs_pmdec', 
        save_folder: str = default_folder):
    """
    Plot Right Ascension (RA) vs Declination (Dec) in proper motion space from a pandas DataFrame.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing at minimum two columns, 'pmra' for Proper Motion in RA
                          and 'pmdec' for Proper Motion in Dec.
        error (bool): If true the plot will include error bars using required columns 'pmra_error', and 'pmdec_error'.
        xlim (int|float, optional): The x-axis upper limit. If None, the default limits are used. Default is None.
        ylim (int|float, optional): The y-axis upper limit. If None, the default limits are used. Default is None.
        color (str, optional): Color of the plotted points. Default is 'red'.
        size (int|float, optional): Size of the plotted points. Default is 0.5.
        title (str, optional): Title of the plot. Default is 'Right Ascension Vs. Declination in Proper Motion Space'.
        save_plot (bool, optional): If true, saves plot as a PDF file. Defaults to False. 
        file_name (str, optional): File name of the resulting plot. Default is 'pmra_vs_pmdec'. File identifier is added automatically.
        save_folder (str, optional): Optional folder destination. A destination folder could also be set using the file name.
    
    Returns:
        None

    Raises:
        TypeError: If the input data is not a pandas DataFrame.
        KeyError: If the required columns are missing ('pmra', 'pmdec')
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Data must be of type pandas.DataFrame')
    # Ensure required columns exist

    required_cols = set()
    if error == False:
        required_cols.update({'pmra', 'pmdec'})
    else:
        required_cols.update({'pmra', 'pmdec', 'pmra_error', 'pmdec_error'})
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame is missing required columns: {', '.join(sorted(missing))}")

    # Proper motion RA, X-Value; Dec, Y-Values
    x = df['pmra']
    y = df['pmdec']
    if error:
        xerr = df['pmra_error']
        yerr = df['pmdec_error']
        plt.errorbar(
            x, y,
            xerr=xerr,
            yerr=yerr,
            fmt='none',
            markersize=size,
            ecolor=color,       # error bar color
            elinewidth=0.5,
            capsize=2,
            alpha=0.4,           # reduce clutter, makes it slightly transparetn
            zorder=1
        )
    plt.scatter(x, y, c=color, s=size, zorder=3)

    #Titles and Show graph
    plt.title(title)
    plt.xlabel("PM RA")
    plt.ylabel("PM Dec")
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if save_plot:
        safe_name = file_name.replace(" ", "_")
        safe_name = f"{safe_name}.pdf"
        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            filepath = os.path.join(save_folder, safe_name)
        else:
            filepath = safe_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filepath}")
    plt.show()

def get_distance(parallax):
    """Convert parallax (mas) to distance (pc)

    Args:
        parallax (float): Parallax in milliarcseconds

    Returns:
        float: Distance in parsecs
    """
    return 1 / (parallax / 1000)


def get_magnitude(phot_g_mean_mag, distance):
    """Convert apparent magnitude to absolute magnitude

    Args:
        phot_g_mean_mag (float): G-band apparent magnitude
        distance (float): Distance in parsecs

    Returns:
        float: Absolute magnitude
    """
    return phot_g_mean_mag - 5 * np.log10(distance / 10)


def get_bprp(phot_bp_mean_mag, phot_rp_mean_mag):
    """Calculate BP-RP colour index

    Args:
        phot_bp_mean_mag (float): BP-band apparent magnitude
        phot_rp_mean_mag (float): RP-band apparent magnitude

    Returns:
        float: BP-RP colour index
    """
    return phot_bp_mean_mag - phot_rp_mean_mag

def G_error(G_flux, G_flux_error):
    G_mag_error = (2.5 / np.log(10)) * (G_flux_error / G_flux)
    return G_mag_error

def G_BP_error(BP_flux, BP_flux_error):
    BP_mag_error = (2.5 / np.log(10)) * (BP_flux_error / BP_flux)
    return BP_mag_error

def G_RP_error(RP_flux, RP_flux_error):
    RP_mag_error = (2.5 / np.log(10)) * (RP_flux_error / RP_flux)
    return RP_mag_error


#This calculates a main sequence line on the Hertzpring Russel Diagram function when the parameter "underlay" is True
_MS_TABLE_X = np.array([p[0] for p in MAIN_SEQUENCE_TABLE])
_MS_TABLE_Y = np.array([p[1] for p in MAIN_SEQUENCE_TABLE])
_MS_INTERPOLATOR = PchipInterpolator(_MS_TABLE_X, _MS_TABLE_Y)

def main_sequence_line(bprp_range=(-0.5, 4.5), n_points=200):
    """Generate an empirical main-sequence reference line.

    Interpolates the tabulated Pecaut & Mamajek (2013, updated) dwarf-star
    sequence of (BP-RP, M_G) points with a monotonic cubic Hermite
    interpolant (PCHIP), so the line follows the real, non-polynomial shape
    of the main sequence (including the steep faintening of the M-dwarf end)
    instead of the overshoot/turnover a single global polynomial fit produces.

    Args:
        bprp_range (tuple): (min, max) BP-RP color range to evaluate over.
                             Clipped to the range actually covered by the
                             reference table (approximately -0.62 to 5.10);
                             a range entirely outside that coverage raises
                             ValueError.
        n_points (int): Number of points along the line.

    Returns:
        (np.ndarray, np.ndarray): (colour, magnitude) arrays for the line.
    """
    lo = max(bprp_range[0], _MS_TABLE_X.min())
    hi = min(bprp_range[1], _MS_TABLE_X.max())
    if lo >= hi:
        raise ValueError(
            f"bprp_range {bprp_range} does not overlap the main-sequence "
            f"reference table's coverage ({_MS_TABLE_X.min():.2f} to "
            f"{_MS_TABLE_X.max():.2f})."
        )
    x = np.linspace(lo, hi, n_points)
    y = _MS_INTERPOLATOR(x)
    return x, y


def plot_hr_diagram(
        df,
        error: bool = True,
        point_colour: str = 'blue',
        underlay: bool = True,
        bg_labels: bool = True,
        sequence: tuple[np.ndarray, np.ndarray] | None = None,
        sequence_label: str = 'Main Sequence',
        sequence_colour: str = 'black',
        xlim: list[int] = [-1, 5],
        ylim: list[int] = [0, 20],
        title: str = "Hertzsprung-Russell Diagram", 
        save_plot: bool = False, 
        file_name: str = "hr_diagram", 
        save_folder: str = default_folder
    ):
    """Plot an HR diagram from a Gaia dataframe.

    Args:
        df (pandas.dataframe): Gaia data containing at minimum parallax, phot_g_mean_mag, phot_bp_mean_mag, and phot_rp_mean_mag.
        error (bool): If true the plot will include error bars using required columns.
        point_colour (str, optional): Colour of the points.
        underlay (bool): If true, overlays a reference sequence line (main sequence by default,
                          or whatever is passed via `sequence`).
        bg_labels (bool): If true, draws approximate, illustrative text labels marking where
                          white dwarfs, giants, and supergiants typically sit on the diagram
                          (see HRD_REGION_LABELS in constants.py). Independent of `underlay`.
        sequence (tuple[array, array], optional): (colour, magnitude) arrays for a custom sequence
                          line (e.g. an isochrone) to draw instead of the built-in main sequence.
                          Only used when `underlay=True`.
        sequence_label (str, optional): Legend label for the overlaid sequence line.
        sequence_colour (str, optional): Colour of the overlaid sequence line.
        xlim ([int]|[float], optional): The x-axis upper limit. If None, the default limits are used. Default is None.
        ylim ([int]|[float], optional): The y-axis upper limit. If None, the default limits are used. Default is None.
        title (str, optional): Title of the plot. Default is 'Hertzsprung-Russell Diagram'.
        save_plot (bool, optional): If true, saves plot as a PDF file. Defaults to False. 
        file_name (str, optional): File name of the resulting plot. Default is 'hr_diagram'. File identifier is added automatically.
        save_folder (str, optional): Optional folder destination. A destination folder could also be set using the file name.

    Returns: 
        None
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError('Data must be of type pandas.DataFrame')
    # Ensure required columns exist
    required_cols = set()
    if error == True:
        required_cols.update({
            "parallax", "parallax_error",
            "phot_g_mean_mag", "phot_g_mean_flux", "phot_g_mean_flux_error",
            "phot_bp_mean_mag", "phot_bp_mean_flux", "phot_bp_mean_flux_error",
            "phot_rp_mean_mag", "phot_rp_mean_flux", "phot_rp_mean_flux_error"
        })
    else:
        required_cols.update({"parallax", "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"})
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame is missing required columns: {', '.join(sorted(missing))}")
    
    mag = []
    colour = []
    mag_err = []
    colour_err = []

    for _, row in df.iterrows():
        M = get_magnitude(row["phot_g_mean_mag"], get_distance(row["parallax"]))
        mag.append(M)
        C = get_bprp(row["phot_bp_mean_mag"], row["phot_rp_mean_mag"])
        colour.append(C)
            
        if error:
            Merr = G_error(row["phot_g_mean_flux"], row["phot_g_mean_flux_error"])
            mag_err.append(Merr)

            BP_err = G_BP_error(row["phot_bp_mean_flux"], row["phot_bp_mean_flux_error"])
            RP_err = G_RP_error(row["phot_rp_mean_flux"], row["phot_rp_mean_flux_error"])
            C_err = np.sqrt(BP_err**2 + RP_err**2)
            colour_err.append(C_err)

    plt.figure()

    if error:
        plt.errorbar(
            colour, mag,
            xerr=colour_err,
            yerr=mag_err,
            fmt='none',
            markersize=1,
            c=point_colour,
            ecolor=point_colour,
            elinewidth=0.5,
            capsize=1,
            alpha=0.4
        )
    else:
        plt.scatter(colour, mag, c=colour, s=1)

    # --- Sequence line overlay ---
    if underlay:
        if sequence is not None:
            seq_x, seq_y = sequence
        else:
            seq_x, seq_y = main_sequence_line(bprp_range=tuple(xlim) if xlim else (-0.5, 4.5))
        plt.plot(seq_x, seq_y, color=sequence_colour, linewidth=1.5,
                  linestyle='--', label=sequence_label, zorder=4)
        plt.legend()

    # --- Approximate region labels (white dwarfs, giants, supergiants) ---
    # Illustrative anchor points only, not precise boundaries or a
    # classification model (see HRD_REGION_LABELS in constants.py for
    # sourcing). Positions are clipped to the current xlim/ylim so a
    # label for a population outside the current view (e.g. supergiants
    # under the default ylim=[0, 20]) is pinned to the nearest edge
    # rather than disappearing off-plot. Controlled independently of
    # `underlay` via `bg_labels`.
    if bg_labels:
        x_bounds = (min(xlim), max(xlim)) if xlim else None
        y_bounds = (min(ylim), max(ylim)) if ylim else None
        for region_label, label_bprp, label_mg in HRD_REGION_LABELS:
            label_x = np.clip(label_bprp, *x_bounds) if x_bounds else label_bprp
            label_y = np.clip(label_mg, *y_bounds) if y_bounds else label_mg
            plt.annotate(
                region_label,
                xy=(label_x, label_y),
                fontsize=9,
                color='dimgray',
                style='italic',
                ha='center',
                va='center',
                zorder=6,
            )

    plt.xlabel(r"G$_{BP}$ - G$_{RP}$")
    plt.ylabel(r"M$_{G}$")
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.title(title)
    plt.gca().invert_yaxis()

    if save_plot:
        safe_name = file_name.replace(" ", "_")
        safe_name = f"{safe_name}.pdf"
        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            filepath = os.path.join(save_folder, safe_name)
        else:
            filepath = safe_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filepath}")
    plt.show()

def hist(
        values, 
        bin_num:int = 50, 
        parallax:bool =False, 
        title:str = "Distances histogram", 
        save_plot: bool = False, 
        file_name: str = "histogram", 
        save_folder: str = default_folder):

    """Plot a histogram.

    Args:
        values (array-like): Values for histogram.
        bin_num (int, optional): Number of bins, defaults to 50.
        parallax (bool, optinal): Set this to be true if using parallax data. If true converts the data into parsecs. 
        title (str, optional): Title of the plot. Default is 'Distances histogram'.
        save_plot (bool, optional): If true, saves plot as a PDF file. Defaults to False. 
        file_name (str, optional): File name of the resulting plot. Default is 'distance_hist'. File identifier is added automatically.
        save_folder (str, optional): Optional folder destination. A destination folder could also be set using the file name.

    Returns: 
        None
    """

    #Adjust if dist given in parallax (convert from mas to parsecs)
    if parallax:
        values = (1000/values)
    
    plt.title(title)
    plt.hist(values, bins=bin_num)
    plt.xlabel('Distance (pc)')
    plt.ylabel('Stars per bin')
    if save_plot:
        safe_name = file_name.replace(" ", "_")
        safe_name = f"{safe_name}.pdf"
        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            filepath = os.path.join(save_folder, safe_name)
        else:
            filepath = safe_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filepath}")
    plt.show()

def gaussian(x, A, sigma, mu):
    return A*(1/(sigma * np.sqrt(2*np.pi)) * np.exp(-1*(x - mu)**2 / (2*sigma**2)))

def fitted_hist(
        values, 
        bin_num:int =50, 
        range:list[int] =[-500,500],
        parallax:bool =False, 
        title:str = "Fitted histogram", 
        save_plot: bool = False, 
        file_name: str = "fitted_hist", 
        save_folder: str = default_folder):
    
    """Plot a fitted histogram.

    Args:
        values (array-like): Values.
        bin_num (int, optional): Number of bins, defaults to 50.
        range (list[int]) X-axis range. Defaults to [-500, 500].
        parallax (bool, optinal): Set this to be true if using parallax data. If true converts the data into parsecs. 
        title (str, optional): Title of the plot. Default is 'Distances histogramm'.
        save_plot (bool, optional): If true, saves plot as a PDF file. Defaults to False. 
        file_name (str, optional): File name of the resulting plot. Default is 'fitted_distance_hist'. File identifier is added automatically.
        save_folder (str, optional): Optional folder destination. A destination folder could also be set using the file name.

    Returns: 
        None
    """
    #Convert if using parallax from mas to parsecs
    if parallax:
        values = (1000/values)

    median = values.median()
    std = values.std()

    print("Median: "+ str(median)+", standard deviation: "+str(std))


    plt.title(title)
    h_1d_output = plt.hist(values, bins=bin_num)
    x_plot = np.linspace(range[0],range[1], 300)
    x_1d_fit = (h_1d_output[1][:-1]+h_1d_output[1][1:])/2
    y_1d_fit = h_1d_output[0]
    fit = curve_fit(gaussian, x_1d_fit, y_1d_fit, p0 = [55, std, median])
    print("Standard Deviation: "+str(std))

    #Fix printing this
    #print(fit)
    plt.plot(x_plot, gaussian(x_plot, *fit[0]), label ='Line of Best Fit')

    plt.xlim(range[0], range[1])
    plt.xlabel('Distance (pc)')
    plt.ylabel('Stars per bin')
    plt.legend()

    if save_plot:
        safe_name = file_name.replace(" ", "_")
        safe_name = f"{safe_name}.pdf"
        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            filepath = os.path.join(save_folder, safe_name)
        else:
            filepath = safe_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filepath}")
    plt.show()