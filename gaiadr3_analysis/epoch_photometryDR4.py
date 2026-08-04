import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PyAstronomy.pyTiming import pyPDM
from .constants import JD_offset
import os


default_folder = "plots"
style = 'seaborn-v0_8-darkgrid'
plt.style.use(style)

def phase(t, T_0, P):
    """
    Compute the phase of time values relative to a reference epoch and period.

    Args:
        t (array-like): Time values.
        T_0 (float): Reference epoch, 't naught' (same units as 't').
        P (float): Period (same units as 't').

    Returns:
        numpy.ndarray: Phase values in the interval (0, 1).
    """
    return ((t-T_0)/P) % 1

#Plot G, Bp and Rp magnitude light curves in time.
def lightcurve(
        df:pd.DataFrame, 
        error:bool = False,
        title:str='Flux Vs. Time', 
        overplot:bool=True, 
        plot_g:bool = True,
        plot_bp:bool = True,
        plot_rp:bool = True,
        rejectflags: bool=False, 
        period:float=None, 
        xlims:tuple[int|float, int|float]=None, 
        ylims:tuple[int|float, int|float]=None, 
        save_plot: bool = False, 
        file_name: str = "lightcurve",
        save_folder: str = default_folder):
    """
    Plot G, Bp and Rp magnitude light curves in time.

    Args:
        df (pd.DataFrame): DataFrame containing photometry and time columns.
        error (bool): If true the plot will include error bars using required columns 'g_transit_flux_error', 'bp_flux_error', 'rp_flux_error'.
        title (str, optional): Plot title. Defaults to 'Flux Vs. Time'.
        overplot (bool, optional): If True, overplot all bands on a single axes. Defaults to True.
        plot_g (bool, optional): If True, includes the G band in the plot. Defaults to True.
        plot_bp (bool, optional): If True, includes the BP band in the plot. Defaults to True.
        plot_rp (bool, optional): If True, includes the RP band in the plot. Defaults to True.
        rejectflags (bool, optional): If True, filter out rows flagged as rejected (uses
            'variability_flag_*_reject' columns). Defaults to False.
        period (float, optional): If provided, fold times on this period (phase plot). Defaults to None.
        xlims (tuple[float, float] or None, optional): X-axis limits. Defaults to None.
        ylims (tuple[float, float] or None, optional): Y-axis limits. Defaults to None.
        save_plot (bool, optional): If true, saves plot as a PDF file. Defaults to False. 
        file_name (str, optional): File name of the resulting plot. Default is "lightcurve". File identifier is added automatically.
        save_folder (str, optional): Optional folder destination. A destination folder could also be set using the file name.

    Raises:
        TypeError: If the input data is not a pandas DataFrame.
        KeyError: If the required columns are missing

    Returns:
        None: Displays a matplotlib figure.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Data must be of type pandas.DataFrame')
    # Ensure required columns exist
    if error == False:
        required_cols = {'g_transit_mag', 'bp_mag', 'rp_mag', 'g_transit_time', 'bp_obs_time', 'rp_obs_time'}
    else:
        required_cols = {'g_transit_mag', 'bp_mag', 'rp_mag', 'g_transit_time', 'bp_obs_time', 'rp_obs_time', 'g_transit_flux', 'bp_flux', 'rp_flux', 'g_transit_flux_error', 'bp_flux_error', 'rp_flux_error'}
    missing = required_cols - set(df.columns)
    
    #Set subplot_num (only applies if over_plot is false)
    subplot_num = 3

    # Only use required variables and columns
    if plot_g == False:
        missing = missing - set('g_transit_mag', 'g_transit_time', 'g_transit_flux', 'g_transit_flux_error')
        subplot_num -= 1
    else:
        g = 'g_transit_mag'
        g_time = 'g_transit_time'

    if plot_bp == False:
        missing = missing - set('bp_mag', 'bp_obs_time', 'bp_flux', 'bp_flux_error')
        subplot_num -= 1
    else:
        bp = 'bp_mag'
        bp_time = 'bp_obs_time'


    if plot_rp == False:
        missing = missing - set('rp_mag', 'rp_obs_time', 'rp_flux', 'rp_flux_error')
        subplot_num -=1
    else:
        rp = 'rp_mag'
        rp_time = 'rp_obs_time'

    # Raise error if columns missing
    if missing:
        raise KeyError(f"DataFrame is missing required columns: {', '.join(sorted(missing))}")

    # Filter Rejections if true
    if rejectflags:
        g_df = df[df['variability_flag_g_reject'] == False]
        bp_df = df[df['variability_flag_bp_reject'] == False]
        rp_df = df[df['variability_flag_rp_reject'] == False]
    else:
        g_df = df
        bp_df = df
        rp_df = df

    print(f"Len g, bp, and rp datasets respectively: {len(g_df)}, {len(bp_df)}, {len(rp_df)}")
    #X-Value: G Transit time
    x_g = g_df[g_time] + JD_offset
    x_bp = bp_df[bp_time] + JD_offset
    x_rp = rp_df[rp_time] + JD_offset
    x_label = "Time (JD)"
    
    #Phase x if true
    if period is not None:
        x_g = phase(x_g, x_g.median(), period)
        x_bp = phase(x_bp, x_g.median(), period)
        x_rp = phase(x_rp, x_g.median(), period)
        x_label = f"Phase, p = {period}"
        print(f"P value: {period}")
    
    #Y-Value: Magnitudes for light band
    y_g = g_df[g] 
    y_bp = bp_df[bp]
    y_rp = rp_df[rp]
    
    #Calculate error values
    if error:
        if plot_g:
            g_err = 1.0857362047581294 * (df['g_transit_flux_error'] / df['g_transit_flux'])
        if plot_bp:
            bp_err = 1.0857362047581294 * (df['bp_flux_error'] / df['bp_flux'])
        if plot_rp:
            rp_err = 1.0857362047581294 * (df['rp_flux_error'] / df['rp_flux'])

    if overplot is True:
        plt.xlabel(x_label)
        plt.ylabel("Band (app mag)")
        if plot_g: 
            if error: plt.errorbar(x_g, y_g, yerr=g_err, fmt='none', ecolor='green', elinewidth=0.5, capsize=3, alpha=0.4)
            plt.scatter(x_g, y_g, c ='green', s = 3, label='G Band')
        if plot_bp: 
            if error: plt.errorbar(x_bp, y_bp, yerr=bp_err, fmt='none', ecolor='blue', elinewidth=0.5, capsize=3, alpha=0.4)
            plt.scatter(x_bp, y_bp, c ='blue', s = 3, label='Bp Band')
        if plot_rp: 
            if error: plt.errorbar(x_rp, y_rp, yerr=rp_err, fmt='none', ecolor='red', elinewidth=0.5, capsize=3, alpha=0.4)
            plt.scatter(x_rp, y_rp, c ='red', s = 3, label='Rp Band')
        plt.title(title)
        plt.legend()
        plt.gca().invert_yaxis()
        if xlims is not None:
            plt.xlim(xlims[0], xlims[1])
        if ylims is not None:
            plt.ylim(ylims[0], ylims[1])
    else:
        fig, axes = plt.subplots(subplot_num, 1, figsize=(6, 10))
        # Plot on each subplot
        #axes[0].set_title(plot_title)
        if plot_g:
            axes[0].set_xlabel(x_label)
            axes[0].set_ylabel("G Band (app mag)")
            if error: axes[0].errorbar(
                    x_g, y_g,
                    yerr=g_err,
                    fmt='none',
                    ecolor='green',       # error bar color
                    elinewidth=0.5,
                    capsize=2,
                    alpha=0.4,           # reduce clutter, makes it slightly transparetn
                )
            axes[0].scatter(x_g, y_g, c ='green', s = 4, label='G Band')
            axes[0].legend()
            axes[0].invert_yaxis()
    
        if plot_bp:
            #axes[1].set_title(plot_title)
            axes[1].set_xlabel(x_label)
            axes[1].set_ylabel("Bp Band (app mag)")
            axes[1].scatter(x_bp, y_bp, c ='blue', s = 4, label='Bp Band')
            axes[1].legend()
            axes[1].invert_yaxis()
    
        if plot_rp:
            #axes[2].set_title(plot_title)
            axes[2].set_xlabel(x_label)
            axes[2].set_ylabel("Rp Band (app mag)")
            axes[2].scatter(x_rp, y_rp, c ='red', s = 4, label='Rp Band')
            axes[2].legend()
            axes[2].invert_yaxis()


        if ylims is not None:
            for ax in axes: 
                #Flipped because the yaxis is inverted.
                ax.set_ylim(ylims[1], ylims[0])
        if xlims is not None:
            for ax in axes: 
                ax.set_ylim(xlims[0], xlims[1])

    plt.tight_layout()

    if save_plot:
        safe_name = file_name.replace(" ", "_")
        safe_name = f"{safe_name}.pdf"
        if default_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            filepath = os.path.join(save_folder, safe_name)
        else:
            filepath = safe_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filepath}")
    plt.show()

#t is times in julian days. If time is given in non-julian date already then jd = False when calling the function.
#t is expected to be df['g_transit_time']
#mag is expected to be df['g_transit_mag']
def lomb_scargle(
    t: pd.DataFrame = None, 
    mag: pd.DataFrame = None, 
    title:str='Lomb-Scargle Periodogram', 
    period_range: list[float] = None, 
    xlims: list[float] = None, 
    jd: bool=True, 
    save_data: bool = False,
    data_file: str = "ls_data", 
    plot:bool=False, 
    save_plot: bool = False,
    plot_file: str = "ls_plot", 
    save_folder: str = default_folder
):
    """
    Compute a Lomb-Scargle periodogram and optionally plot the result.

    Args:
        t (array-like): Time values (JD or relative). If None, synthetic data is used as an example.
        mag (array-like): Magnitudes or fluxes corresponding to 't'. If None, synthetic data is used as an example.
        title (str, optional): Display title for the corresponding plot. Defaults to 'Lomb-Scargle Periodogram'.
        period_range (list[float], optional): [P_min, P_max] search range in days. If None, it is estimated using the Nyquist frequency.
        xlims (list[float], optional): X-axis limits for period plot (days).
        jd (bool, optional): If True, convert JD times to relative by subtracting the minimum. Defaults to True.
        save_data (bool, optional): If true, saves plot as a CSV file. Defaults to False. 
        data_file (str, optional): File name of the resulting data. Default is "ls_data". File identifier is added automatically.
        plot (bool, optional): If True, display the Lomb-Scargle analysis and an inset zoom around the best period. Defaults to False.
        save_plot (bool, optional): If true, saves plot as a PDF file. Defaults to False. 
        plot_file (str, optional): File name of the resulting plot. Default is "ls_plot". File identifier is added automatically.
        save_folder (str, optional): Optional folder destination. A destination folder could also be set using the file name.


    Returns:
        pandas.DataFrame(): A pandas data frame containing to following columns: ["period", "power", "false alarm probability"].
    """
    # Default example for if t & y are not inputted.
    if t is None or mag is None:
        rand = np.random.default_rng(42)
        #Time
        t = 100 * rand.random(100)
        #Frequency
        mag = np.sin(2 * np.pi * t) + 0.1 * rand.standard_normal(100)

    #Convert Julian Days
    if jd:
        t = t - t.min()

    #Frequency Range

    # If user didn't specify frequency_range, set it automatically
    if period_range is None:
        # Compute Nyquist period limit
        dt_median = np.median(np.diff(np.sort(t)))
        f_nyq = 1 / (2 * dt_median)
        P_min = 1 / f_nyq
        P_max = t.max() - t.min()
        period_range = [P_min, P_max]

        
    ls = LombScargle(t, mag)
    frequency, power = ls.autopower(
        minimum_frequency=1/period_range[1],    # max period
        maximum_frequency=1/period_range[0]  # min period
    )
    
    #Convert frequency to period
    period_days = 1/frequency

    #False Alarm Probabilities
    FAP = [ls.false_alarm_probability(p) for p in power]

    df = pd.DataFrame({"period":period_days, "power":power, "false alarm probability":FAP})

    if plot:
        plot_ls(period_days=period_days, power=power, title=title, xlims=xlims, save=save_plot, file_name=plot_file, save_folder=save_folder)
    if save_data:
        os.makedirs(save_folder, exist_ok=True)
        safe_name = data_file.replace(" ", "_")
        safe_name = f"{safe_name}.csv"
        if default_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            filepath = os.path.join(save_folder, safe_name)
        else:
            filepath = safe_name
        df.to_csv(filepath)
        print(f"Data saved as {filepath}")

    #return data
    return (df)


def plot_ls(
        period_days:pd.DataFrame, 
        power:pd.DataFrame, 
        title:str, 
        xlims=None, 
        save:bool = False, 
        file_name:str="ls_plot", 
        save_folder: str = default_folder
    ):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(period_days, power)
    ax.set_xlabel("Period (days)")
    if(xlims is not None):
        ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylabel("Lomb-Scargle Power")
    ax.grid(True)
    plt.title(title)

    #Create sub inset graph of best period
    sub_ax = inset_axes(
        parent_axes=ax,
        width="30%",
        height="30%",
        borderpad=2
    )

    # Define zoom window (+-10% around best period)
    period_hours = period_days * 24
    best_idx = np.argmax(power)
    best_period_days = period_days[best_idx]
    best_period_hours = best_period_days * 24
    print(f"Best period: {best_period_days:.6f} days ({best_period_hours:.3f} hours)")

    zoom_width = 0.10 * best_period_hours
    mask = (period_hours > best_period_hours - zoom_width) & (period_hours < best_period_hours + zoom_width)
    sub_ax.plot(period_hours[mask], power[mask])
    sub_ax.set_xlabel("Period (hours)")
    sub_ax.grid(True)
    
    if save:
        safe_name = file_name.replace(" ", "_")
        safe_name = f"{safe_name}.pdf"
        if default_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            filepath = os.path.join(save_folder, safe_name)
        else:
            filepath = safe_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filepath}")
    plt.show()

def pdm(
        t: pd.DataFrame, 
        mag: pd.DataFrame, 
        title:str='Phase Dispersion Minimization', 
        bins:int|float = 50, 
        covers:int = 3, 
        freq_range:list[int|float] = [0.01, 10.0, 0.001], 
        plot = False, 
        save_data: bool = False,
        data_file: str = "pdm_data",
        save_plot: bool = False,
        plot_file: str = "pdm_plot", 
        save_folder: str = default_folder
    ):
    """
    Compute a Phase Dispersion Minimization and optionally plot the result.

    Args:
        t (array-like): Time values (JD or relative).
        mag (array-like): Magnitudes or fluxes corresponding to 't'. 
        title (str, optional): Display title for the corresponding plot. Defaults to 'Phase Dispersion Minimization'.
        bins (int, optional): Number of bins to be used in the PDM analysis. Defaults to 50.
        covers (int, optional): Number of covers to be uesd in the PDM analysis. Defaults to 3.
        freq_range (list[float], optional): Frequency range of the PDM analysis, defaults to [0.01, 10.0, 0.001], 
            where the third value is increment.
        plot (bool, optional): If True, display the PDM analysis as a plot. Defaults to False.
        save_data (bool, optional): If true, saves plot as a CSV file. Defaults to False. 
        data_file (str, optional): File name of the resulting data. Default is "pdm_data". File identifier is added automatically.
        save_plot (bool, optional): If true, saves plot as a PDF file. Defaults to False. 
        plot_file (str, optional): File name of the resulting plot. Default is "pdm_plot". File identifier is added automatically.
        save_folder (str, optional): Optional folder destination. A destination folder could also be set using the file name.

    Returns:
        pandas.DataFrame(): A pandas data frame containing to following columns: ["period", "frequency", "theta"].
    """
    # Define trial frequencies
    fmin = freq_range[0]
    fmax = freq_range[1]
    dfreq = freq_range[2]
    
    S = pyPDM.Scanner(minVal=fmin, maxVal=fmax, dVal=dfreq, mode="frequency")
    P = pyPDM.PyPDM(t, mag)

    frequencies, theta = P.pdmEquiBinCover(
        bins,     # number of bins
        covers,      # covers
        S
    )

    best_frequency = frequencies[np.argmin(theta)]
    best_period = 1.0 / best_frequency
    periods = [(1.0/f) for f in frequencies]

    print("Best period =", best_period, "days")

    if plot == True:
        plot_pdm(frequencies=frequencies, theta=theta, best_period=best_period, save=save_plot, title=title, file_name=plot_file, save_folder=save_folder)
    
    df = pd.DataFrame({"period":periods, "frequency":frequencies, "theta":theta})

    if save_data:
        os.makedirs(save_folder, exist_ok=True)
        safe_name = data_file.replace(" ", "_")
        safe_name = f"{safe_name}.csv"
        if default_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            filepath = os.path.join(save_folder, safe_name)
        else:
            filepath = safe_name
        df.to_csv(filepath)
        print(f"Data saved as {filepath}")
    
    return (df)

def plot_pdm(frequencies, theta, best_period:float = None, save:bool=False, title:str= "PDM Plot", file_name:str="pdm_plot", save_folder: str = default_folder):
    plt.figure(figsize=(8,5))
    plt.plot(1/frequencies, theta, 'k-')
    if(best_period is not None):
        plt.axvline(best_period, c='red', label=f"Best Period: {best_period:6f} days")
    plt.xlabel("Period (days)")
    plt.ylabel("Theta")
    plt.legend()
    plt.title(title)

    if save:
        safe_name = file_name.replace(" ", "_")
        safe_name = f"{safe_name}.pdf"
        if default_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            filepath = os.path.join(save_folder, safe_name)
        else:
            filepath = safe_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filepath}")
    plt.show()