import os
import importlib

import pandas

import transport_simulation
from transport_simulation import TransportSimulation
from matplotlib import pyplot as plt
import numpy as np
import map_param_grid
import multiprocess
import matplotlib as mpl
import pickle
import pandas as pd
import seaborn as sns
from matplotlib import ticker
import copy
import importlib
import stats_grid
import time


def create_plot(plot, plot_data: list, xlabel=None, ylabel=None, set_lsandfs: bool = False):
    line_colors = ['bo', 'ro']
    for i in range(len(plot_data)):
        line = plot_data[i]
        plot.plots(line['x_data'], line['y_data'], line_colors[i], label=line['label'], markersize=4)
    ylabel is not None and plot.set_ylabel(ylabel)
    xlabel is not None and plot.set_xlabel(xlabel)
    if len(plot_data) > 1:
        plot.legend(frameon=True)
    if set_lsandfs:
        positions = [0.5, 1.0, 2.0, 4.0]
        plot.yaxis.set_major_locator(ticker.FixedLocator(positions))
        plot.yaxis.set_major_formatter(ticker.FixedFormatter(positions))
        plot.yaxis.set_minor_locator(ticker.FixedLocator([]))
        plot.yaxis.set_minor_formatter(ticker.FixedFormatter([]))


def plot_simulation_attributes(stats, attributes_list, log: bool = True):
    """


    :param stats:
    :param attributes_list:
    :param log:
    """
    def merge_lists(lists):
        return [x for y in lists for x in y]

    attributes = merge_lists(attributes_list)
    assert (len(attributes) == sum([len(x) for x in attributes_list]))
    fig, axes = plt.subplots(len(attributes_list), 1, figsize=(10, 15), squeeze=False)
    x = stats['time_sec']
    for cur_attributes, ax in zip(attributes_list, axes[:, 0]):
        print(cur_attributes)
        for attr in cur_attributes:
            ax.plot(x, stats[attr], label=attr)
        if log:
            ax.set_yscale('log')
        ax.set_xlabel("time [sec]")
        ax.set_ylabel("nmol")
        ax.legend()
        if is_log:
            ylim = ax.get_ylim()
            ylim = (10.0, ylim[1])
            ax.set_ylim(ylim)
    return axes


def get_ts_time_series(dt_sec: float, **kwargs) -> transport_simulation.TransportSimulation:
    """


    :param dt_sec: time step for the simulation (in seconds)
    :param kwargs: transport simulation parameters and their desired values (some parameters may be overridden later in
                   this function)
    :return: transport simulation with the appropriate parameters
    """
    ts = TransportSimulation(v_N_L=627e-15, v_C_L=2194e-15, **kwargs)
    ts.bleach_start_time_sec = 400.0
    ts.set_time_step(dt_sec)
    ts.set_NPC_dock_sites(n_NPCs=2000, n_dock_sites_per_NPC=500)
    ts.set_passive_nuclear_molar_rate_per_sec(0.02)
    ts.set_params(fraction_complex_NPC_traverse_per_sec=100, rate_free_to_complex_per_sec=0.05)
    ts.bleach_volume_L_per_sec = 100.0e-15
    #ts.rate_complex_to_NPC_per_free_site_per_sec_per_M= 1.0e+6
    ts.fraction_complex_NPC_to_free_N_per_M_GTP_per_sec = 0.05e+6  # TODO: this is doubled relative to complex_N to free_N
    ts.fraction_complex_N_to_free_N_per_M_GTP_per_sec = 0.05e+6

    return ts


def get_ts_time_series_2(passive_nuclear_molar_rate_per_sec: float, is_force: bool,
                         **kwargs) -> transport_simulation.TransportSimulation:
    """


    :param passive_nuclear_molar_rate_per_sec: the max rate of passive diffusion in 1/(second*M)
    :param is_force: whether the cell nucleus is experiencing force (cell volumes increase)
    :param kwargs: transport simulation parameters and their desired values
    :return: transport simulation with the appropriate parameters
    """
    v_N_L, v_C_L = (762e-15, 4768e-15) if is_force else (627e-15, 2194e-15)
    ts = transport_simulation.TransportSimulation(v_N_L=v_N_L, v_C_L=v_C_L)
    ts.set_time_step(1.0e-3)
    ts.set_NPC_dock_sites(n_NPCs=2000, n_dock_sites_per_NPC=500)
    ts.set_passive_nuclear_molar_rate_per_sec(
        passive_nuclear_molar_rate_per_sec)  #get_passive_export_rate_per_sec(27,1))
    ts.fraction_complex_NPC_to_free_N_per_M_GTP_per_sec = 10.0e+6  # TODO: this is doubled relative to complex_N to free_N
    ts.fraction_complex_N_to_free_N_per_M_GTP_per_sec = 10.0e+6
    ts.rate_complex_to_NPC_per_free_site_per_sec_per_M = 50.0e+6
    ts.fraction_complex_NPC_to_complex_N_C_per_sec = 25.0  # Leakage parameter
    ts.rate_GDP_N_to_GTP_N_per_sec = 200.0
    ts.rate_GTP_N_to_GDP_N_per_sec = 2.0
    ts.rate_GTP_C_to_GDP_C_per_sec = 500.0
    ts.rate_GTP_N_to_GTP_C_per_sec = 25.0
    ts.rate_GDP_C_to_GDP_N_per_sec = 3.0
    ts.rate_GDP_N_to_GDP_C_per_sec = 3.0
    ts.rate_complex_to_free_per_sec = 0.05
    ts.rate_free_to_complex_per_sec = 0.05
    ts.fraction_complex_NPC_traverse_per_sec = 200
    ts.set_params(**kwargs)  # override defaults
    return ts


def get_ts_time_series_3(passive_nuclear_molar_rate_per_sec: float, is_force: bool, dt_sec: float = 0.3e-3,
                         **kwargs) -> transport_simulation.TransportSimulation:
    """


    :param passive_nuclear_molar_rate_per_sec: the max rate of passive diffusion in 1/(second*M)
    :param is_force: whether the cell nucleus is experiencing force (cell volumes increase)
    :param dt_sec: time step for the simulation (in seconds)
    :param kwargs: transport simulation parameters and their desired values
    :return: transport simulation with the appropriate parameters
    """
    v_N_L, v_C_L = (762e-15, 4768e-15) if is_force else (627e-15, 2194e-15)
    ts = transport_simulation.TransportSimulation(v_N_L=v_N_L, v_C_L=v_C_L)
    ts.set_time_step(dt_sec)
    ts.set_NPC_dock_sites(n_NPCs=2000, n_dock_sites_per_NPC=500)
    ts.set_passive_nuclear_molar_rate_per_sec(
        passive_nuclear_molar_rate_per_sec)  #get_passive_export_rate_per_sec(27,1))
    ts.fraction_complex_NPC_to_free_N_per_M_GTP_per_sec = 1.0e+6  # TODO: this is doubled relative to complex_N to free_N
    ts.fraction_complex_N_to_free_N_per_M_GTP_per_sec = 1.0e+6
    ts.rate_complex_to_NPC_per_free_site_per_sec_per_M = 50e+6
    ts.fraction_complex_NPC_to_complex_N_C_per_sec = 1000.0  # Leakage parameter
    ts.rate_GDP_N_to_GTP_N_per_sec = 1000.0
    ts.rate_GTP_N_to_GDP_N_per_sec = 0.2
    ts.rate_GTP_C_to_GDP_C_per_sec = 500.0
    ts.rate_GTP_N_to_GTP_C_per_sec = 0.5
    ts.rate_GDP_C_to_GDP_N_per_sec = 1.0
    ts.rate_GDP_N_to_GDP_C_per_sec = 1.0
    ts.rate_complex_to_free_per_sec = 0.05
    ts.rate_free_to_complex_per_sec = 0.01
    ts.fraction_complex_NPC_traverse_per_sec = 4000
    ts.set_params(**kwargs)  # override defaults
    return ts


def get_param_range_D_kon(nx: int, ny: int) -> dict:
    """


    :param nx:
    :param ny:
    :return:
    """
    param_range = {}
    print(f"nx={nx} ny={ny}")
    epsilon = 1e-9
    v_N_L = 627e-15
    param_range['tag_x'] = "max_passive_diffusion_rate_nmol_per_sec_per_M"
    param_range['range_x'] = np.logspace(-4, 0,
                                         nx) * transport_simulation.N_A * v_N_L  # divided to convert from nuclear passive diffusion rate r, where dN/dt = r*([C]-[N]))
    param_range['pretty_x'] = r"passive diffusion rate [$s^{-1} M^{-1}$]"
    param_range['tag_y'] = "rate_free_to_complex_per_sec"
    param_range['range_y'] = np.logspace(-4, 0, ny)
    param_range['pretty_y'] = r"NTR $k_{on}$ [$sec^{-1}$]"
    return param_range


def get_transport_simulation_map_passive(**kwargs) -> transport_simulation.TransportSimulation:
    """


    :param kwargs: transport simulation parameters and their desired values
    :return: transport simulation with the appropriate parameters
    """
    ts = transport_simulation.TransportSimulation(**kwargs)
    ts.set_time_step(1e-3)
    ts.set_v_N_L(627e-15, True)
    ts.set_v_C_L(2194e-15, True)
    ts.set_NPC_dock_sites(n_NPCs=2000, n_dock_sites_per_NPC=500)
    ts.fraction_complex_NPC_to_free_N_per_M_GTP_per_sec = 0.5e+6  # TODO: this is doubled relative to complex_N to free_N
    ts.fraction_complex_N_to_free_N_per_M_GTP_per_sec = 0.5e+6
    return ts


def get_param_range_traverse_kon(nx: int, ny: int) -> dict:
    """
    Obtains the specific NPC traverse rate values and free-to-complex rate values to iterate over

    :param nx: number of values used to discretize npc_traverse_range
    :param ny: number of values used to discretize k_on_range
    :return: dictionary containing information on the discrete NPC traverse rates (x-axis) and free-to-complex rates
             (y-axis) to be modelled
    """
    return stats_grid.get_param_range_traverse_kon(nx, ny, npc_traverse_range=(1, 1000), k_on_range=(0.01, 10))


def get_transport_simulation_by_passive(passive_nuclear_molar_rate_per_sec: float, is_force_volume: bool, Ran_cell_M: float = 20.0e-6,
                                        **kwargs):
    """


    :param passive_nuclear_molar_rate_per_sec: the max rate of passive diffusion in 1/(second*M)
    :param is_force_volume: whether to use the volumes for when the cell nucleus is experiencing force
    :param Ran_cell_M: total Ran concentration in the nucleus and cytoplasm combined (in M)
    :param kwargs: transport simulation parameters and their desired values
    :return: transport simulation with the appropriate parameters
    """
    v_N_L, v_C_L = (762e-15, 4768e-15) if is_force_volume else (627e-15, 2194e-15)
    ts = transport_simulation.TransportSimulation(v_N_L=v_N_L, v_C_L=v_C_L)
    ts.set_time_step(0.1e-3)
    ts.set_NPC_dock_sites(n_NPCs=2000, n_dock_sites_per_NPC=500)
    ts.fraction_complex_NPC_to_free_N_per_M_GTP_per_sec = 1.0e+6  # TODO: this is doubled relative to complex_N to free_N
    ts.fraction_complex_N_to_free_N_per_M_GTP_per_sec = 1.0e+6
    ts.set_passive_nuclear_molar_rate_per_sec(
        passive_nuclear_molar_rate_per_sec)  #get_passive_export_rate_per_sec(27,1))
    ts.rate_complex_to_NPC_per_free_site_per_sec_per_M = 50e+6
    ts.fraction_complex_NPC_to_complex_N_C_per_sec = 3000.0  # Leakage parameter
    ts.rate_GDP_N_to_GTP_N_per_sec = 1000.0
    ts.rate_GTP_N_to_GDP_N_per_sec = 0.2
    ts.rate_GTP_C_to_GDP_C_per_sec = 500.0
    ts.rate_GTP_N_to_GTP_C_per_sec = 0.5
    ts.rate_GDP_C_to_GDP_N_per_sec = 1.0
    ts.rate_GDP_N_to_GDP_C_per_sec = 1.0
    ts.rate_complex_to_free_per_sec = 0.05
    ts.rate_free_to_complex_per_sec = 0.01  # SCAN
    ts.fraction_complex_NPC_traverse_per_sec = 4000  # SCAN
    ts.set_params(**kwargs)  # override defaults
    ts.set_RAN_distribution(Ran_cell_M=Ran_cell_M,
                            # total physiological concentration of Ran # TODO: check in the literature
                            parts_GTP_N=1000,
                            parts_GTP_C=1,
                            parts_GDP_N=1,
                            parts_GDP_C=1000)
    return ts


def plot_stats_grids(stats_grids, transport_simulation, NC_min=1.0, NC_max=20.0, vmax_import_export=10.0) -> None:
    """
    Calls stats_grid.py to create different contour plots (e.g. N/C ratio, bound fraction for nucleus and cytoplasm,
    nuclear import and export rate plots, and import/export ratio plots)

    :param stats_grids: dictionary containing species information at the end of each simulation in the form of numpy
                        arrays (where each j-i pair is the result of a particular y-x pairing)
    :param transport_simulation: transport simulation instance used to obtain nuclear and cytoplasmic volumes
    :param NC_min: min value for the N/C ratio axis
    :param NC_max: max value for the N/C ratio axis
    :param vmax_import_export: max value for the nuclear import and export rate plots
    :return: None
    """
    stats_grid.plot_stats_grids(stats_grids, transport_simulation, param_range, NC_min, NC_max, vmax_import_export)


def get_passive_nuclear_molar_rate_per_sec(MW: int, is_force: bool) -> float:  # TODO: verify it corresponds to multiplyng by concentration rather than nmolecules
    """
    Obtains the maximum passive rate of nuclear diffusion given a cargo's molecular weight and whether the nucleus is
    experiencing force

    :param MW: molecular weight of the cargo molecule (in kDa)
    :param is_force: whether the cell nucleus is experiencing force
    :return: the max rate of passive diffusion in 1/(second*M)
    """
    import pdb; pdb.set_trace()
    base_rates = {27: 0.0805618,
                  41: 0.06022355,
                  54: 0.03301662,
                  67: 0.0287649}
    rate = base_rates[MW]
    if is_force:
        rate += get_force_effect_on_diffusion(MW)
    return rate


def get_force_effect_on_diffusion(MW: int) -> float:
    """
    Obtains the increase in passive diffusion rate as an effect of nuclear force for a given cargo molecular weight
    (as measured by experiment)

    :param MW: molecular weight of the cargo molecule (in kDa)
    :return: the increase to max rate of passive diffusion in 1/(second*M)
    """
    effects = {27: 0.08214946,
               41: 0.03027974,
               54: 0.01,  # 54:0.00026308,
               67: 0.01}  #67:0.00272423 }
    return effects[MW]


def my_plot_param_grid(df: pandas.DataFrame,  # a pivoted 2D dataframe
                       pretty_x=None, pretty_y=None, pretty_z=None, is_colorbar=False, **contourf_kwargs):
    """


    :param df:
    :param pretty_x:
    :param pretty_y:
    :param pretty_z:
    :param is_colorbar:
    :param contourf_kwargs:
    :return:
    """
    X, Y = np.meshgrid(df.columns, df.index)
    ax = plt.gca()
    ctr = plt.contourf(X, Y, df.to_numpy(), **contourf_kwargs)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(pretty_x)
    ax.set_ylabel(pretty_y)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if xlim[1] > xlim[0]:
        ax.set_xlim(xlim[1], xlim[0])
    if is_colorbar:
        cb = plt.colorbar(label=pretty_z)
        ticks = cb.get_ticks()
        cb.set_ticks(ticks)
        cb.set_ticklabels(["{:.2f}".format(tick) for tick in ticks])
    return ctr


def transform_N2C_to_N_relative(N2C, N2C_vol):
    """

    :param N2C:
    :param N2C_vol:
    :return:
    """
    import pdb; pdb.set_trace()
    N_relative = (N2C * N2C_vol) / (1 + N2C * N2C_vol)
    return N_relative


def plot_dX_dY_Z(df: pandas.DataFrame, is_transform: bool = False, N2C_vol: float = 0.333) -> None:
    """


    :param df:
    :param is_transform:
    :param N2C_vol:
    :return: None
    """
    NLSs = df['rate_free_to_complex_per_sec'].unique()
    NLS_starti = len(NLSs) // 4
    NLS_endi = len(NLSs) // 4 * 3
    NLSs = NLSs[NLS_starti:NLS_endi]
    nrow = len(NLSs)
    ncol = 3 if is_transform else 4
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=False, squeeze=False,
                             figsize=(4.5 * ncol + 1.0, 3.2 * nrow + 1.0))
    if is_transform:
        zlabel_by_column = {0: r"dN / dX",
                            1: r"dN / dY",
                            2: r"N saturation",
                            3: r"N saturation"}
    else:
        zlabel_by_column = {0: r"dNC / dX",
                            1: r"dNC / dY",
                            2: r"N:C ratio",
                            3: r"N:C ratio"}

    for axes, NLS in zip(axes, NLSs):
        is_NLS = np.isclose(df['rate_free_to_complex_per_sec'], NLS)
        N2C = df[is_NLS].pivot(index='passive_rate', columns='fraction_complex_NPC_traverse_per_sec', values='N2C')
        N_relative = (N2C * N2C_vol) / (1 + N2C * N2C_vol)
        for column, ax in enumerate(axes):
            if column in [0, 1]:

                if is_transform:
                    is_linear = True
                    Z_diff = np.diff(N_relative.values, axis=(1 - column), append=0)
                    Z = pd.DataFrame(Z_diff, index=N_relative.index, columns=N_relative.columns)
                    NC_min = 0.0
                    NC_max = 0.05 if column == 0 else 0.01
                else:
                    is_linear = False
                    Z_diff = np.diff(N2C.values, axis=(1 - column), append=0)
                    Z = pd.DataFrame(np.abs(Z_diff) / N2C.values, index=N2C.index, columns=N2C.columns)
                    NC_min = 0.01
                    NC_max = 0.4
            else:
                is_linear = False
                if is_transform:
                    Z = N_relative
                    NC_min = 0.05
                    NC_max = 1.0
                    #levels= np.linspace(NC_min, NC_max, 11)
                    #locator= None
                else:
                    Z = N2C
                    NC_min = 1.0
                    NC_max = 10.0 if column == 2 else 50.0
            if is_linear:
                levels = np.logspace(NC_min, NC_max, 11)
                locator = None
            else:
                levels = np.logspace(np.log2(NC_min), np.log2(NC_max), 11, base=2.0)
                locator = mpl.ticker.LogLocator(base=2.0)
            plt.sca(ax)
            ct = my_plot_param_grid(Z,
                                    pretty_x=r'rate NPC traverse [$sec^{-1}$]',
                                    pretty_y=r'passive rate [$sec^{-1}$]' if column == 0 else None,
                                    pretty_z=zlabel_by_column[column],
                                    is_colorbar=not is_shared_colorbar,
                                    vmin=NC_min,
                                    vmax=NC_max,
                                    levels=levels,
                                    locator=locator,
                                    extend='both')
            colors = ['r', 'g', 'b', 'k']
            ylim = ax.get_ylim()
            for MW, color in zip(MWs, colors):
                y0 = get_passive_nuclear_molar_rate_per_sec(MW, is_force=False)
                y1 = get_passive_nuclear_molar_rate_per_sec(MW, is_force=True)
                y1 = min(y1, ylim[1] * 0.99)
                x0 = N2C.columns[0]
                x1 = N2C.columns[-1]
                ax.plot((x0, x1), (y0, y0), color + "-", label=str(MW))
                plt.annotate(str(MW), (x0, y0), color=color)
                ax.plot((x0, x1), (y1, y1), color + "-", linewidth=3.0)
            ax.set_ylim(ylim)
            plt.title(f"NLS $k_{{on}}$={NLS * 1000:.1f} $ms^{{-1}}$")
    plt.tight_layout()


def plot_cells(
        cells_column: str = 'fraction_complex_NPC_traverse_per_sec',
        pretty_cell: str = "rate NPC traverse {:.1f} $sec^{{-1}}$",
        x_column: str = 'passive_rate',
        pretty_x: str = r'passive rate [$sec^{-1}$]',
        y_column: str = 'rate_free_to_complex_per_sec',
        pretty_y: str = r'NLS $k_{on}$ [$sec^{-1}$]') -> None:
    """


    :param cells_column:
    :param pretty_cell:
    :param x_column:
    :param pretty_x:
    :param y_column:
    :param pretty_y:
    :return: None
    """
    plt.style.use('./my.mplstyle.txt')
    cells = df[cells_column].unique()[:2]
    ncol = 3
    nrow = len(traverses) // ncol + (len(traverses) % ncol > 0) * 1
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True, squeeze=False, figsize=(17.0, 4.0 * nrow + 1.0))
    axes_iter = iter(axes.reshape(-1))
    for cell in cells:
        ax = next(axes_iter)
        is_cell = np.isclose(df[cells_column], cell)
        N2C = df[is_cell].pivot(index=y_column, columns=x_column, values='N2C')
        NC_min = 1.0
        NC_max = 10.0
        plt.sca(ax)
        ct = my_plot_param_grid(N2C,
                                pretty_x=pretty_x,
                                pretty_y=pretty_y,
                                pretty_z=r'N:C',
                                vmin=NC_min,
                                vmax=NC_max,
                                levels=np.logspace(np.log2(NC_min), np.log2(NC_max), 21, base=2.0),
                                locator=mpl.ticker.LogLocator(base=2.0),
                                extend='both')
        colors = ['r', 'g', 'b', 'k']
        if x_column == "passive_rate":
            xlim = ax.get_xlim()
            for MW, color in zip(MWs, colors):
                x0 = get_passive_nuclear_molar_rate_per_sec(MW, is_force=False)
                x1 = get_passive_nuclear_molar_rate_per_sec(MW, is_force=True)
                x1 = min(x1, xlim[1] * 0.99)
                y0 = N2C.index[0]
                y1 = N2C.index[-1]
                ax.plot((x0, x0), (y0, y1), color + "-", label=str(MW))
                plt.annotate(str(MW), (x0, y0), color=color)
                ax.plot((x1, x1), (y0, y1), color + "-", linewidth=3.0)
            ax.set_xlim(xlim)
        if y_column == "passive_rate":
            pass
        plt.title(pretty_cell.format(cell))
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.3, 0.03, 0.4])
    cb = fig.colorbar(ct, cax=cbar_ax)
    ticks = cb.get_ticks()
    cb.set_ticks(ticks)
    cb.set_ticklabels(["{:.2f}".format(tick) for tick in ticks])


def get_dLogRatios_by_passive(stats_grids_by_passive, ts_by_passive) -> dict:
    """


    :param stats_grids_by_passive:
    :param ts_by_passive:
    :return:
    """
    import pdb; pdb.set_trace()
    dRatios_by_passive = {}
    keys = sorted(stats_grids_by_passive.keys())
    for key0, key1 in zip(keys[:-1], keys[1:]):
        stats_grids = [stats_grids_by_passive[key0], stats_grids_by_passive[key1]]
        ts = [ts_by_passive[key0], ts_by_passive[key1]]
        v_N_L = ts[0].get_v_N_L()
        v_C_L = ts[0].get_v_C_L()
        assert v_N_L == ts[1].get_v_N_L()
        assert v_C_L == ts[1].get_v_C_L()
        ratios0 = map_param_grid.get_N_to_C_ratios(stats_grids[0], v_N_L, v_C_L)
        ratios1 = map_param_grid.get_N_to_C_ratios(stats_grids[1], v_N_L, v_C_L)
        dRatios_by_passive[key0] = np.log(ratios1) - np.log(ratios0)
    return dRatios_by_passive


def get_ts_with_parameters(MW: int = 27, NLS_strength: int = 0, is_force: bool = False,
                           is_change_cell_volume: bool = False, **kwargs) -> transport_simulation.TransportSimulation:
    """
    Creates a

    :param MW: molecular weight of the cargo molecule (in kDa)
    :param NLS_strength: relative strength of the nuclear localization signal
    :param is_force: whether the cell nucleus is experiencing force
    :param is_change_cell_volume: whether the cell volumes change when force is applied
    :return: transport simulation instance with the appropriate specs
    """
    v_N_L, v_C_L = (762e-15, 4768e-15) if is_force and is_change_cell_volume else (627e-15, 2194e-15)
    ts = transport_simulation.TransportSimulation(v_N_L=v_N_L, v_C_L=v_C_L)
    ts.set_time_step(0.1e-3)
    ts.set_NPC_dock_sites(n_NPCs=2000, n_dock_sites_per_NPC=500)
    ts.fraction_complex_NPC_to_free_N_per_M_GTP_per_sec = 1.0e+6  # TODO: this is doubled relative to complex_N to free_N
    ts.fraction_complex_N_to_free_N_per_M_GTP_per_sec = 1.0e+6
    ts.rate_complex_to_NPC_per_free_site_per_sec_per_M = 50e+6
    ts.fraction_complex_NPC_to_complex_N_C_per_sec = 3000.0  # Leakage parameter
    ts.rate_GDP_N_to_GTP_N_per_sec = 1000.0
    ts.rate_GTP_N_to_GDP_N_per_sec = 0.2
    ts.rate_GTP_C_to_GDP_C_per_sec = 500.0
    ts.rate_GTP_N_to_GTP_C_per_sec = 0.5
    ts.rate_GDP_C_to_GDP_N_per_sec = 1.0
    ts.rate_GDP_N_to_GDP_C_per_sec = 1.0
    ts.rate_complex_to_free_per_sec = 0.05
    #
    ts.set_passive_nuclear_molar_rate_per_sec(
        get_passive_nuclear_molar_rate_per_sec(MW, is_force))
    ts.rate_free_to_complex_per_sec = get_free_to_complex_rate(NLS_strength)
    ts.fraction_complex_NPC_traverse_per_sec = get_fraction_complex_NPC_traverse_per_sec(MW, is_force)
    #
    ts.set_params(**kwargs)  # override defaults
    return ts


def get_free_to_complex_rate(NLS_strength: int) -> float:
    """
    Obtains the free cargo to complexed cargo conversion rate given an NLS strength

    :param NLS_strength: relative strength of the nuclear localization sequence
    :return float: rate of free cargo conversion to complexed cargo (in 1/sec)
    """
    return free_to_complex_rates[NLS_strength]


def get_fraction_complex_NPC_traverse_per_sec(MW: int, is_force: bool) -> float:
    """
    Calculates the fraction of complexes that go from one side of the NPC to the other per sec given the molecular
    weight of the cargo and whether a force is being applied

    :param MW: molecular weight of the cargo molecule (in kDa)
    :param is_force: whether the cell nucleus is experiencing force
    :return: the fraction of complexes that traverse the NPC (in 1/sec)
    """
    no_force = 30.0
    force = 200.0
    rate_row = [no_force, force]
    rate = {mw: rate_row for mw in [27, 41, 54, 67]}
    #        27: [30.0,  200.0],
    #        41: [30.0,  200.0],
    #        54: [30.0, 200.0],
    #        67: [30.0,  s*10.0] }
    i_force = 1 if is_force else 0
    return rate[MW][i_force]


def get_compartment_nmol_stats(ts: transport_simulation.TransportSimulation, stats: dict, compartment: str,
                               labels: list = ['L', 'U']):
    """

    :param ts: the transport simulation that the stats are from
    :param stats:
    :param compartment: the area to get the stats for (e.g. nucleus, cytoplasm, or NPC)
    :param labels: the list of types of cargo to include (labeled and/or unlabeled)
    :return:
    """
    import pdb; pdb.set_trace()
    assert (compartment in ['N', 'C', 'NPC'])
    nframes = len(stats['time_sec'])
    nmol_stats = np.zeros(nframes)
    if compartment == 'NPC':
        for label in labels:
            for side in ['N', 'C']:
                for source in ['import', 'export']:
                    tag = 'complex{}_NPC_{}_{}'.format(label, side, source)
                    nmol_stats = nmol_stats + stats[tag]
    else:
        for state in ['free', 'complex']:
            for label in labels:
                tag = '{}{}_{}'.format(state,
                                       label,
                                       compartment)
                nmol_stats = nmol_stats + stats[tag]
    return nmol_stats


def get_compartment_concentration_stats(ts: transport_simulation.TransportSimulation, stats, compartment: str,
                                        labels: list = ['L', 'U']):
    """

    :param ts:
    :param stats:
    :param compartment: the area to get the stats for (e.g. nucleus, cytoplasm, or NPC)
    :param labels: the list of types of cargo to include (labeled and/or unlabeled)
    :return: the molar concentration of cargo in a given compartment (in M)
    """
    import pdb; pdb.set_trace()
    assert (compartment in ['N', 'C'])
    nmol_stats = get_compartment_nmol_stats(ts,
                                            stats,
                                            compartment,
                                            labels)
    is_nuclear = (compartment == 'N')
    volume_L = (ts.get_v_N_L() if is_nuclear else ts.get_v_C_L())
    return (nmol_stats / transport_simulation.N_A) / volume_L


def get_N_C_ratio_stats(ts: transport_simulation.TransportSimulation, stats, labels: list = ['L', 'U']):
    """

    :param ts:
    :param stats:
    :param labels: the list of types of cargo to include (labeled and/or unlabeled)
    :return:
    """
    EPSILON = 1E-12
    c_N_stats = get_compartment_concentration_stats(ts,
                                                    stats,
                                                    'N')
    c_C_stats = get_compartment_concentration_stats(ts,
                                                    stats,
                                                    'C')
    x = map_param_grid.get_N_to_C_ratios_for_ts(stats, ts)
    import pdb; pdb.set_trace()
    print(x == c_N_stats / c_C_stats)
    return c_N_stats / c_C_stats


def do_simulate(ts: transport_simulation.TransportSimulation, simulation_time_sec: float):
    """
    Simulate the transport simulation for certain amount of time

    :param ts: the transport simulation instance
    :param simulation_time_sec: time interval (in seconds) for which the simulation is run
    :return: dictionary with that contain the values for each species at each time frame
    """
    return ts.simulate(simulation_time_sec)


def get_MW_stats_list_by_force(MW: int, simulation_time_sec: float, n_processors: int,
                               is_change_cell_volume: bool = False):
    """


    :param MW: molecular weight of the cargo (in kDa)
    :param simulation_time_sec: time interval (in seconds) for which the simulation is run
    :param n_processors: number of CPUs to use for multiprocessing
    :param is_change_cell_volume: whether the cell volume changes when force is applied
    """
    import pdb; pdb.set_trace()
    assert (MW in [27, 41, 54, 67])
    stats_list_by_force = {}
    TSs_by_force = {}
    for is_force in [False, True]:
        TS_tuples = []
        for i_NLS in range(len(free_to_complex_rates)):
            ts = get_ts_with_parameters(MW=MW,
                                        NLS_strength=i_NLS,
                                        is_force=is_force,
                                        is_change_cell_volume=is_change_cell_volume)
            TS_tuples.append((ts, simulation_time_sec))
        pool = multiprocess.Pool(processes=n_processors)
        stats_list_by_force[is_force] = pool.starmap(do_simulate, TS_tuples)
        TSs_by_force[is_force] = [x[0] for x in TS_tuples]
        print(f"Is force {is_force} i_NLS {i_NLS}: OK")
    return stats_list_by_force, TSs_by_force


def plot_MW_stats_list(stats_list_by_force, TSs_by_force) -> None:
    """

    :param stats_list_by_force:
    :param TSs_by_force:
    :return: None
    """
    plot_from_sec = 0.1  # ts.bleach_start_time_sec + 1.0
    extras = [  #'GTP_N',
        #'GDP_N',
        #'GTP_C',
        #'GDP_C',
        #'complexL_C',
        #'freeL_C',
        #'complexL_N',
        #'freeL_N'
    ]
    fig, ax_grid = plt.subplots(7 + len(extras), 3,
                                figsize=(15,
                                         40.0 + 5.0 * len(extras)),
                                sharex=False, sharey=False)
    n_NLS = len(stats_list_by_force[False])
    assert (n_NLS == len(stats_list_by_force[True]))
    ratios = np.ones(shape=(7 + len(extras), n_NLS))
    ax_grid = ax_grid.transpose()
    for axes, is_force in zip(ax_grid[0:2, :], [False, True]):
        for i_NLS, stats in enumerate(stats_list_by_force[is_force]):
            ts = TSs_by_force[is_force][i_NLS]
            labels = ['L', 'U']
            x = stats['time_sec']
            ys = {
                0: stats['nuclear_importL_per_sec'] + stats['nuclear_importU_per_sec'],
                1: stats['nuclear_exportL_per_sec'] + stats['nuclear_exportU_per_sec'],
                2: get_N_C_ratio_stats(ts, stats, labels),
                3: get_compartment_concentration_stats(ts, stats, 'C', labels),
                4: get_compartment_concentration_stats(ts, stats, 'N', labels),
                6: stats['complexL_NPC_N_import'] + stats['complexL_NPC_C_import'] + stats['complexL_NPC_N_export'] +
                   stats['complexL_NPC_C_export'] + stats['complexU_NPC_C_import'] + stats['complexU_NPC_C_import'] +
                   stats['complexU_NPC_N_export'] + stats['complexU_NPC_C_export']
            }
            ys[5] = ys[3] - ys[4]
            for iextra, extra in enumerate(extras):
                ys[7 + iextra] = stats[extra]
            plot_from_frame = int(plot_from_sec / ts.dt_sec)
            for i_row, ax in enumerate(axes):
                ax.plot(x[plot_from_frame:], ys[i_row][plot_from_frame:], label=get_free_to_complex_rate(i_NLS))
                ax.set_xlabel(r"time [$sec$]")
                if is_force:
                    ratios[i_row, i_NLS] *= ys[i_row][-1]
                else:
                    ratios[i_row, i_NLS] /= ys[i_row][-1]
            axes[0].set_ylabel(r"import rate [$sec^{-1}$]")
            axes[0].set_ylim([0.01, 0.3])
            #axes[0].set_yscale('log')
            axes[1].set_ylabel(r"export rate [$sec^{-1}$]")
            axes[1].set_ylim([0.01, 0.3])
            #axes[1].set_yscale('log')
            axes[2].set_ylabel("N/C ratio")
            axes[2].set_ylim([0, 7.0])
            axes[3].set_ylabel(r"C [$M$]")
            axes[3].set_yscale('log')
            axes[4].set_ylabel(r"N [$M$]")
            axes[4].set_yscale('log')
            axes[5].set_ylabel(r"$\Delta$(C,N) [$M$]")
            axes[5].set_yscale('symlog', linthresh=1e-9)
            axes[6].set_ylabel('NPC [nmol]')
            for iextra, extra in enumerate(extras):
                axes[7 + iextra].set_ylabel(extra)
                axes[7 + iextra].set_yscale('log')
            title = "30 kPa" if is_force else "5 kPa"
            axes[0].set_title(title)

    NLSs = [get_free_to_complex_rate(i_NLS) for i_NLS in range(ratios.shape[1])]
    ax_grid[2, 0].set_title("Mechanosensitivity")
    for i_row, ax in enumerate(ax_grid[2, :]):
        ax.bar(range(len(NLSs)), ratios[i_row, :], width=0.8, tick_label=NLSs)
        ax.set_xlabel('NLS strength')

    handles, labels = ax_grid[0, 0].get_legend_handles_labels()
    lh = fig.legend(handles, labels, loc='center left')
    lh.set_title('NLS strength')


if __name__ == '__main__':

    # Unit testing

    #  CELL 2:
    #NOTE: Most test runs are on a conda py36 environment on Mac
    # In principle, should work on any python but multiprocessing packaage
    # is sensitive and may require tuning.
    os.system('pwd')


    #  CELL 3:
    os.system('python test_transport_simulation.py')


    # Model variables - legend:
    # complexL/U - importin-cargo complex, labeled (L) or unlabeled (U)
    #
    # freeL/U - free cargo, labeled (L) or unlabeled (U)
    #
    # c - concentration
    #
    # v - volume
    #
    # C - cytoplasm
    #
    # N - nucleues
    #
    # NPC - nuclear pore complex
    #
    # nmol - number of molecules
    #
    # M - molar (moles per liter)
    #
    # L - liter
    #
    # fL - femtoliter

    #  CELL 6:
    importlib.reload(transport_simulation)

    # Simulation main code


    #  CELL 11:
    # RUN:
    """
    sim_time_sec = 50.0
    sim_flags = {}  #rate_free_to_complex_per_sec=1.0,
    #max_passive_diffusion_rate_nmol_per_sec_per_M=2e7)
    #ts= get_ts_time_series(dt_sec= 2e-3, **sim_flags)
    ts = get_ts_time_series_3(passive_nuclear_molar_rate_per_sec=0.04, is_force=False, **sim_flags)
    stats = ts.simulate(sim_time_sec, nskip_statistics=10)


    #  CELL 12:
    ####
    RAN_attributes = ['GDP_N', 'GDP_C', 'GTP_N', 'GTP_C']
    cargoL_attributes = ['complexL_NPC_C_import', 'complexL_NPC_C_export',
                         'complexL_NPC_N_import', 'complexL_NPC_N_export',
                         'freeL_N', 'freeL_C', 'complexL_C', 'complexL_N']
    cargoU_attributes = ['complexU_NPC_C_import', 'complexU_NPC_C_export',
                         'complexU_NPC_N_import', 'complexU_NPC_N_export',
                         'freeU_N', 'freeU_C', 'complexU_C', 'complexU_N']
    c_attributes = ["c_C_M", "c_C_M"]
    b_attributes = ['fraction_C_b', "c_C_M", ]
    npc_attributes = ['nmol_NPC']
    dock_attributes = ['c_C_M', 'fraction_C_b', 'nmol_NPC', 'NPC_dock_capacity']
    is_log = True
    ax = plot_simulation_attributes(
        stats,
        [RAN_attributes,
         cargoL_attributes,
         cargoU_attributes],
        log=is_log)


    #  CELL 13:
    ## plt.figure()
    plt.plot(stats['time_sec'],
             (stats['freeL_N'] + stats['complexL_N'] + stats['freeU_N'] + stats['complexU_N'])
             / (stats['freeL_C'] + stats['complexL_C'] + stats['freeU_C'] + stats['complexU_C']) * (
                         ts.get_v_C_L() / ts.get_v_N_L()))
    plt.xlabel(r'time [$s$]')
    plt.ylabel(r'N:C')
    #plt.ylim(0,5)
    plt.figure()
    for tag in ['import', 'export']:
        plt.plot(stats['time_sec'], stats[f'nuclear_{tag}L_per_sec'], label=tag)
        plt.yscale('log')
        plt.legend()


    #  CELL 14:
    ### Slow/normal
    ts_slow = get_ts_time_series(2e-3,
                                 rate_GDP_N_to_GTP_N_per_sec=0.2)
    #fig, ax = plt.subplots(1, 3,figsize=(40,5))
    stats_slow = ts_slow.simulate(sim_time_sec)
    ax = plot_simulation_attributes(stats_slow, [cargoL_attributes], log=is_log)
    plt.title("slow GDP_N to GTP_N rate")
    ts_normal = get_ts_time_series(2e-3)
    stats_normal = ts_normal.simulate(sim_time_sec)
    ax = plot_simulation_attributes(stats_normal, [cargoL_attributes], log=is_log)
    plt.title("normal GDP_N to GTP_N rate")

    ## Map parameters phasespace of transport


    #  CELL 21:
    importlib.reload(map_param_grid)

    n_processors = os.cpu_count()
    param_range = get_param_range_D_kon(nx=4, ny=4)
    print("*** Starting multiprocess run ***")
    stats_grids_passive, ts_passive = map_param_grid.map_param_grid_parallel(param_range,
                                                                             equilibration_time_sec=600.0,
                                                                             n_processors=n_processors,
                                                                             transport_simulation_generator=get_transport_simulation_map_passive)
    print("*** Finished multiprocess run ***")


    #  CELL 22:
    param_range2 = param_range.copy()
    param_range2['pretty_x'] = r"dN/dt passive rate [$s^{-1}$]"
    param_range2['range_x'] = param_range['range_x'] / transport_simulation.N_A / ts_passive.get_v_N_L()
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.5), sharex=True, sharey=True)
    NC_min = 1.0
    NC_max = 8.0
    # N/C
    plt.sca(axes[0, 0])
    map_param_grid.plot_NC_ratios(param_range2, stats_grids_passive, ts_passive, vmin=NC_min, vmax=NC_max)
    # Bound fraction
    map_param_grid.plot_bound_fraction(param_range2, stats_grids_passive, 'N', ax=axes[0, 1])
    map_param_grid.plot_bound_fraction(param_range2, stats_grids_passive, 'C', ax=axes[0, 2])
    # Import/export
    map_param_grid.plot_import_export(param_range2, stats_grids_passive, axes=[axes[1, 0], axes[1, 1]])
    plt.sca(axes[1, 2])
    ratios_import_export = map_param_grid.get_import_export_ratios(stats_grids_passive)
    map_param_grid.plot_param_grid(param_range2, ratios_import_export, Z_label='import:export', vmin=NC_min,
                                   vmax=NC_max, levels=np.linspace(NC_min, NC_max, 21), extend='both')
    print("End of cell 22")
    plt.show()
    time.sleep(5)
    print(param_range)
    print(param_range2)

    ## map traverse


    #  CELL 24:
    importlib.reload(map_param_grid)
    importlib.reload(transport_simulation)


    #  CELL 26:
    test_ts = get_transport_simulation_by_passive(0.02, False)
    print(test_ts.max_passive_diffusion_rate_nmol_per_sec_per_M)


    #  CELL 27:
    importlib.reload(map_param_grid)

    mpl.rc('image', cmap='RdYlBu')

    #  CELL 28:
    # TIME CONSUMING #
    param_range = get_param_range_traverse_kon(nx=10, ny=10)
    print(param_range)
    n_processors = multiprocess.cpu_count()
    stats_grids_traverse_by_passive_force = {}  # 2D maps of statistics for different passive diffusion params
    ts_traverse_by_passive_force = {}  # transport simulaiton object used for each
    print("*** Starting multiprocess run ***")
    from tqdm.auto import tqdm
    passive_space = np.logspace(np.log10(0.005), np.log10(0.15), 15) # 0.01,0.09,6)
    for i in tqdm(range(len(passive_space))):
        passive = passive_space[i]
        for is_force_volume in [False]:  # [False, True]
            def transport_simulation_generator(**kwargs):
                return get_transport_simulation_by_passive(passive_nuclear_molar_rate_per_sec=passive,
                                                           is_force_volume=is_force_volume,
                                                           **kwargs)


            key = (passive, is_force_volume)
            stats_grids_traverse_by_passive_force[key], \
                ts_traverse_by_passive_force[key] = \
                map_param_grid.map_param_grid_parallel(param_range,
                                                       equilibration_time_sec=100.0,
                                                       n_processors=n_processors - 3,
                                                       transport_simulation_generator=transport_simulation_generator)
            print(f"passive rate {passive} is old force_volume {is_force_volume}")
            plot_stats_grids(stats_grids_traverse_by_passive_force[key],
                             ts_traverse_by_passive_force[key],
                             vmax_import_export=10.0,
                             NC_max=30.0,
                             NC_min=1.0)
    print("End of cell 28")
    plt.show()
    time.sleep(5)
    print("*** Finished multiprocess run ***")
    # Pickle results

    with open("./Results/Heatmaps_Ran20uM.pkl", "wb") as F:
        pickle.dump([stats_grids_traverse_by_passive_force, ts_traverse_by_passive_force], F)


    #  CELL 30:
    keys = sorted(stats_grids_traverse_by_passive_force.keys(), key=lambda x: (x[0], x[1]))
    for key in keys:
        if key[1]:
            continue
        print(f"passive rate {key[0]} is force {key[1]}")
        plot_stats_grids(stats_grids_traverse_by_passive_force[key], ts_traverse_by_passive_force[key],
                         vmax_import_export=10.0, NC_max=3.0, NC_min=1.0)
    print("End of cell 30")
    plt.show()
    time.sleep(5)

    #  CELL 31:
    keys = sorted(stats_grids_traverse_by_passive_force.keys(), key=lambda x: (x[0], x[1]))
    for key in keys:
        if key[1]:
            continue
        print(f"passive rate {key[0]} is force {key[1]}")
        plot_stats_grids(stats_grids_traverse_by_passive_force[key], ts_traverse_by_passive_force[key],
                         vmax_import_export=10.0, NC_max=30.0, NC_min=1.0)
    print("End of cell 31")
    plt.show()
    time.sleep(5)

    ## Heatmaps with Ran of 40 mM


    #  CELL 33:
    # TIME CONSUMING #
    Ran_cell_M = 40.0e-6
    param_range = get_param_range_traverse_kon(nx=7, ny=7)
    print(param_range)
    n_processors = multiprocess.cpu_count()
    stats_grids_traverse_by_passive_force = {}  # 2D maps of statistics for different passive diffusion params
    ts_traverse_by_passive_force = {}  #  transport simulaiton object used for each
    print("*** Starting multiprocess run ***")
    from tqdm.auto import tqdm
    passive_space = np.logspace(np.log10(0.01), np.log10(0.15), 15)  # 0.01,0.09,6)
    for i in tqdm(range(len(passive_space))):
        passive = passive_space[i]
        for is_force_volume in [False]:  # [False, True]
            def transport_simulation_generator(**kwargs):
                print(f"Ran: {Ran_cell_M:.6f} M")
                return get_transport_simulation_by_passive(passive_nuclear_molar_rate_per_sec=passive,
                                                           is_force_volume=is_force_volume,
                                                           Ran_cell_M=Ran_cell_M,
                                                           **kwargs)


            key = (passive, is_force_volume)
            stats_grids_traverse_by_passive_force[key], \
                ts_traverse_by_passive_force[key] = \
                map_param_grid.map_param_grid_parallel(param_range,
                                                       equilibration_time_sec=100.0,
                                                       n_processors=n_processors // 2,
                                                       transport_simulation_generator=transport_simulation_generator)
            print(f"passive rate {passive} is old force_volume {is_force_volume}")
            plot_stats_grids(stats_grids_traverse_by_passive_force[key],
                             ts_traverse_by_passive_force[key],
                             vmax_import_export=10.0,
                             NC_max=30.0,
                             NC_min=1.0)
    print("End of cell 33")
    plt.show()
    time.sleep(5)
    print("*** Finished multiprocess run ***")

    # Pickle results

    with open(f"./Results/Heatmaps_Ran{(1E6 * Ran_cell_M):.0f}uM.pkl", "wb") as F:
        pickle.dump([stats_grids_traverse_by_passive_force, ts_traverse_by_passive_force], F)
    os.system('ls Results')

    ## Heatmaps with Ran of 80 mM


    #  CELL 36:
    # TIME CONSUMING #
    Ran_cell_M = 80.0e-6
    param_range = get_param_range_traverse_kon(nx=7, ny=7)
    print(param_range)
    n_processors = multiprocess.cpu_count()
    CACHE = True
    if not CACHE:
        stats_grids_traverse_by_passive_force = {}  # 2D maps of statistics for different passive diffusion params
        ts_traverse_by_passive_force = {}  #  transport simulaiton object used for each
    print("*** Starting multiprocess run ***")
    from tqdm.auto import tqdm
    passive_space = np.logspace(np.log10(0.01), np.log10(0.1), 10) # 0.01,0.09,6)
    for i in tqdm(range(len(passive_space))):
        passive = passive_space[i]
        for is_force_volume in [False]:  # [False, True]
            def transport_simulation_generator(**kwargs):
                print(f"Ran: {Ran_cell_M:.6f} M")
                return get_transport_simulation_by_passive(passive_nuclear_molar_rate_per_sec=passive,
                                                           is_force_volume=is_force_volume,
                                                           Ran_cell_M=Ran_cell_M,
                                                           **kwargs)


            key = (passive, is_force_volume)
            if key in stats_grids_traverse_by_passive_force:
                continue
            stats_grids_traverse_by_passive_force[key], \
                ts_traverse_by_passive_force[key] = \
                map_param_grid.map_param_grid_parallel(param_range, equilibration_time_sec=100.0,
                                                       n_processors=n_processors // 2,
                                                       transport_simulation_generator=transport_simulation_generator)
            print(f"passive rate {passive} is old force_volume {is_force_volume}")
            plot_stats_grids(stats_grids_traverse_by_passive_force[key], ts_traverse_by_passive_force[key],
                             vmax_import_export=10.0, NC_max=30.0, NC_min=1.0)
    print("End of cell 36")
    plt.show()
    time.sleep(5)
    print("*** Finished multiprocess run ***")
    # Pickle results

    with open(f"./Results/Heatmaps_Ran{(1E6 * Ran_cell_M):.0f}uM.pkl", "wb") as F:
        pickle.dump([stats_grids_traverse_by_passive_force, ts_traverse_by_passive_force], F)
    os.system('ls Results')

    ## Run 03 on cluster (~/raveh_lab/Projects/npctransport_kinetics/run03/)

    """
    #  CELL 38:
    importlib.reload(map_param_grid)

    with open("run03/run03.pkl", "rb") as F:
        stats_grids_by_passive, ts_by_passive, param_range = pickle.load(F)

    print([f'{state}{label}_N' for state in ['free', 'complex'] for label in ['L', 'U']])
    print(stats_grids_by_passive.keys())

    #  CELL 39:
    if False:
        # TIME CONSUMING #
        Ran_cell_M = 80.0e-6
        param_range = get_param_range_traverse_kon(nx=7, ny=7)
        print(param_range)
        n_processors = multiprocess.cpu_count()
        CACHE = True
        if CACHE:
            stats_grids_traverse_by_passive_force = {}  # 2D maps of statistics for different passive diffusion params
            ts_traverse_by_passive_force = {}  #  transport simulaiton object used for each
        print("*** Starting multiprocess run ***")
        for passive in np.logspace(np.log10(0.01), np.log10(0.1), 10):  #0.01,0.09,6):
            for is_force_volume in [False]:  # [False, True]
                def transport_simulation_generator(**kwargs):
                    print(f"Ran: {Ran_cell_M:.6f} M")
                    return get_transport_simulation_by_passive(passive_nuclear_molar_rate_per_sec=passive,
                                                               is_force_volume=is_force_volume,
                                                               Ran_cell_M=Ran_cell_M,
                                                               **kwargs)


                key = (passive, is_force_volume)
                if key in stats_grids_traverse_by_passive_force:
                    continue
                stats_grids_traverse_by_passive_force[key], \
                    ts_traverse_by_passive_force[key] = \
                    map_param_grid.map_param_grid_parallel(param_range, equilibration_time_sec=100.0,
                                                           n_processors=n_processors // 2,
                                                           transport_simulation_generator=transport_simulation_generator)
                print(f"passive rate {passive} is old force_volume {is_force_volume}")
                plot_stats_grids(stats_grids_traverse_by_passive_force[key], ts_traverse_by_passive_force[key],
                                 vmax_import_export=10.0, NC_max=30.0, NC_min=1.0)
        print("*** Finished multiprocess run ***")
        # Pickle results

        with open(f"./Results/Heatmaps_Ran{(1E6 * Ran_cell_M):.0f}uM.pkl", "wb") as F:
            pickle.dump( [stats_grids_traverse_by_passive_force, ts_traverse_by_passive_force], F)
        os.system('ls Results')


    #  CELL 40:
    pd.set_option("display.max_columns", None)
    df = map_param_grid.get_df_from_stats_grids_by_passive \
        (param_range, stats_grids_by_passive, ts_by_passive)
    print(df.head())


    #  CELL 42:
    ###### VISUALIZATION OF RESULTS ######
    MWs = [27, 41, 54, 67]


    #  CELL 43:
    """
    plt.style.use('./my.mplstyle.txt')
    NLSs = df['passive_rate'].unique()[:]
    ncol = 3
    nrow = len(NLSs) // ncol + (len(NLSs) % ncol > 0) * 1
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True, squeeze=False, figsize=(17.0, 4.0 * nrow + 1.0))
    axes_iter = iter(axes.reshape(-1))
    import pdb; pdb.set_trace()
    for NLS in NLSs:
        ax = next(axes_iter)
        is_NLS = np.isclose(df['passive_rate'], NLS)
        N2C = df[is_NLS].pivot(index='passive_rate', columns='fraction_complex_NPC_traverse_per_sec', values='N2C')
        if N2C.empty:
            continue
        NC_min = 1.0
        NC_max = 10.0
        plt.sca(ax)
        ct = my_plot_param_grid(N2C, pretty_x=r'rate NPC traverse [$sec^{-1}$]', pretty_y=r'passive rate [$sec^{-1}$]',
                                pretty_z=r'N:C', vmin=NC_min, vmax=NC_max,
                                levels=np.logspace(np.log2(NC_min), np.log2(NC_max), 21, base=2.0),
                                locator=mpl.ticker.LogLocator(base=2.0), extend='both')
        colors = ['r', 'g', 'b', 'k']
        ylim = ax.get_ylim()
        for MW, color in zip(MWs, colors):
            y0 = get_passive_nuclear_molar_rate_per_sec(MW, is_force=False)
            y1 = get_passive_nuclear_molar_rate_per_sec(MW, is_force=True)
            y1 = min(y1, ylim[1] * 0.99)
            x0 = N2C.columns[0]
            x1 = N2C.columns[-1]
            ax.plot((x0, x1), (y0, y0), color + "-", label=str(MW))
            plt.annotate(str(MW), (x0, y0), color=color)
            ax.plot((x0, x1), (y1, y1), color + "-", linewidth=3.0)
        ax.set_ylim(ylim)
        plt.title(f"NLS $k_{{on}}${NLS:.4f} $sec^{{-1}}$")
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.3, 0.03, 0.4])
    cb = fig.colorbar(ct, cax=cbar_ax)
    ticks = cb.get_ticks()
    cb.set_ticks(ticks)
    cb.set_ticklabels(["{:.2f}".format(tick) for tick in ticks])
    print("End of cell 43")
    plt.show()
    """


    #  CELL 44:
    plt.style.use('./my.mplstyle.txt')
    NLSs = df['rate_free_to_complex_per_sec'].unique()[:]
    ncol = 3
    nrow = len(NLSs) // ncol + (len(NLSs) % ncol > 0) * 1
    fig, axes = plt.subplots(nrow, ncol,
                             sharex=True,
                             sharey=True,
                             squeeze=False,
                             figsize=(17.0, 4.0 * nrow + 1.0))
    axes_iter = iter(axes.reshape(-1))
    for NLS in NLSs:
        ax = next(axes_iter)
        is_NLS = np.isclose(df['rate_free_to_complex_per_sec'], NLS)
        N2C = df[is_NLS].pivot(index='passive_rate',
                               columns='fraction_complex_NPC_traverse_per_sec',
                               values='N2C')
        NC_min = 1.0
        NC_max = 10.0
        plt.sca(ax)
        ct = my_plot_param_grid(N2C,
                                pretty_x=r'rate NPC traverse [$sec^{-1}$]',
                                pretty_y=r'passive rate [$sec^{-1}$]',
                                pretty_z=r'N:C',
                                vmin=NC_min,
                                vmax=NC_max,
                                levels=np.logspace(np.log2(NC_min), np.log2(NC_max), 21, base=2.0),
                                locator=mpl.ticker.LogLocator(base=2.0),
                                extend='both')
        colors = ['r', 'g', 'b', 'k']
        ylim = ax.get_ylim()
        for MW, color in zip(MWs, colors):
            y0 = get_passive_nuclear_molar_rate_per_sec(MW, is_force=False)
            y1 = get_passive_nuclear_molar_rate_per_sec(MW, is_force=True)
            y1 = min(y1, ylim[1] * 0.99)
            x0 = N2C.columns[0]
            x1 = N2C.columns[-1]
            ax.plot((x0, x1), (y0, y0), color + "-", label=str(MW))
            plt.annotate(str(MW), (x0, y0), color=color)
            ax.plot((x0, x1), (y1, y1), color + "-", linewidth=3.0)
        ax.set_ylim(ylim)
        plt.title(f"NLS $k_{{on}}${NLS:.4f} $sec^{{-1}}$")
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.3, 0.03, 0.4])
    cb = fig.colorbar(ct, cax=cbar_ax)
    ticks = cb.get_ticks()
    cb.set_ticks(ticks)
    cb.set_ticklabels(["{:.2f}".format(tick) for tick in ticks])
    print("End of cell 44")
    plt.show()

    ## absolute dZ/dY (normalized by Z)


    #  CELL 46:
    plt.style.use('./my.mplstyle.txt')
    NLSs = df['rate_free_to_complex_per_sec'].unique()[:]
    ncol = 3
    nrow = len(NLSs) // ncol + (len(NLSs) % ncol > 0) * 1
    fig, axes = plt.subplots(nrow, ncol,
                             sharex=True,
                             sharey=True,
                             squeeze=False,
                             figsize=(17.0, 4.0 * nrow + 1.0))
    axes_iter = iter(axes.reshape(-1))
    is_shared_colorbar = True
    for NLS in NLSs:
        ax = next(axes_iter)
        is_NLS = np.isclose(df['rate_free_to_complex_per_sec'], NLS)
        N2C = df[is_NLS].pivot(index='passive_rate',
                               columns='fraction_complex_NPC_traverse_per_sec',
                               values='N2C')
        N2C_diff = pd.DataFrame(np.abs(np.diff(N2C.values, axis=0, append=0) / N2C.values),
                                index=N2C.index,
                                columns=N2C.columns)
        NC_min = 0.01
        NC_max = 0.4
        #    levels= np.logspace(-np.log2(NC_max),-np.log2(NC_min),21, base=2.0),
        #                                 locator= mpl.ticker.LogLocator(base=2.0),
        plt.sca(ax)
        ct = my_plot_param_grid(N2C_diff,
                                pretty_x=r'rate NPC traverse [$sec^{-1}$]',
                                pretty_y=r'passive rate [$sec^{-1}$]',
                                pretty_z=r'N:C',
                                is_colorbar=not is_shared_colorbar,
                                vmin=NC_min,
                                vmax=NC_max,
                                levels=np.logspace(np.log2(NC_min), np.log2(NC_max), 21, base=2.0),
                                locator=mpl.ticker.LogLocator(base=2.0),
                                extend='both')
        colors = ['r', 'g', 'b', 'k']
        ylim = ax.get_ylim()
        for MW, color in zip(MWs, colors):
            y0 = get_passive_nuclear_molar_rate_per_sec(MW, is_force=False)
            y1 = get_passive_nuclear_molar_rate_per_sec(MW, is_force=True)
            y1 = min(y1, ylim[1] * 0.99)
            x0 = N2C.columns[0]
            x1 = N2C.columns[-1]
            ax.plot((x0, x1), (y0, y0), color + "-", label=str(MW))
            plt.annotate(str(MW), (x0, y0), color=color)
            ax.plot((x0, x1), (y1, y1), color + "-", linewidth=3.0)
        ax.set_ylim(ylim)
        plt.title(f"NLS $k_{{on}}${NLS:.4f} $sec^{{-1}}$")
    if is_shared_colorbar:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.3, 0.03, 0.4])
        cb = fig.colorbar(ct, cax=cbar_ax)
        ticks = cb.get_ticks()
        cb.set_ticks(ticks)
        cb.set_ticklabels(["{:.2f}".format(tick) for tick in ticks])
    print("End of cell 46")
    plt.show()


    #  CELL 47:
    ## absolute dZ/dX (normalized by Z)


    #  CELL 48:
    plt.style.use('./my.mplstyle.txt')
    NLSs = df['rate_free_to_complex_per_sec'].unique()[:]
    ncol = 3
    nrow = len(NLSs) // ncol + (len(NLSs) % ncol > 0) * 1
    fig, axes = plt.subplots(nrow, ncol,
                             sharex=True,
                             sharey=True,
                             squeeze=False,
                             figsize=(17.0, 4.0 * nrow + 1.0))
    axes_iter = iter(axes.reshape(-1))
    is_shared_colorbar = False
    for NLS in NLSs:
        ax = next(axes_iter)
        is_NLS = np.isclose(df['rate_free_to_complex_per_sec'], NLS)
        N2C = df[is_NLS].pivot(index='passive_rate',
                               columns='fraction_complex_NPC_traverse_per_sec',
                               values='N2C')
        N2C_diff = pd.DataFrame(np.abs(np.diff(N2C.values, axis=1, append=0)) / N2C.values,
                                index=N2C.index,
                                columns=N2C.columns)
        NC_min = 0.01
        NC_max = 0.4
        #    levels= np.logspace(-np.log2(NC_max),-np.log2(NC_min),21, base=2.0),
        #                                 locator= mpl.ticker.LogLocator(base=2.0),
        plt.sca(ax)
        ct = my_plot_param_grid(N2C_diff,
                                pretty_x=r'rate NPC traverse [$sec^{-1}$]',
                                pretty_y=r'passive rate [$sec^{-1}$]',
                                pretty_z=r'N:C',
                                is_colorbar=not is_shared_colorbar,
                                vmin=NC_min,
                                vmax=NC_max,
                                levels=np.logspace(np.log2(NC_min), np.log2(NC_max), 21, base=2.0),
                                locator=mpl.ticker.LogLocator(base=2.0),
                                extend='both')
        colors = ['r', 'g', 'b', 'k']
        ylim = ax.get_ylim()
        for MW, color in zip(MWs, colors):
            y0 = get_passive_nuclear_molar_rate_per_sec(MW, is_force=False)
            y1 = get_passive_nuclear_molar_rate_per_sec(MW, is_force=True)
            y1 = min(y1, ylim[1] * 0.99)
            x0 = N2C.columns[0]
            x1 = N2C.columns[-1]
            ax.plot((x0, x1), (y0, y0), color + "-", label=str(MW))
            plt.annotate(str(MW), (x0, y0), color=color)
            ax.plot((x0, x1), (y1, y1), color + "-", linewidth=3.0)
        ax.set_ylim(ylim)
        plt.title(f"NLS $k_{{on}}${NLS:.4f} $sec^{{-1}}$")
    if is_shared_colorbar:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.3, 0.03, 0.4])
        cb = fig.colorbar(ct, cax=cbar_ax)
        ticks = cb.get_ticks()
        cb.set_ticks(ticks)
        cb.set_ticklabels(["{:.2f}".format(tick) for tick in ticks])
    print("End of cell 48")
    plt.show()

    # Get precise numbers for figures (Jul 8, 2021)

    ## no transform


    #  CELL 51:
    VERBOSE = False
    plt.style.use('./my.mplstyle.txt')
    NLSs = df['rate_free_to_complex_per_sec'].unique()
    if VERBOSE or True:
        print("NLSs [s^-1]: ", NLSs)
    NLSs_subset = np.array([1.00000000e-03, 1.30494682e-03, 1.70288620e-03, 2.22217592e-03,
                            2.89982140e-03, 3.78411271e-03, 4.93806584e-03, 6.44391330e-03,
                            8.40896415e-03, 1.09732510e-02, 1.43195090e-02, 1.86861977e-02,
                            2.43844942e-02, 3.18204681e-02, 4.15240186e-02, 5.41866359e-02,
                            7.07106781e-02, 9.22736744e-02, 1.20412238e-01, 1.57131566e-01,
                            2.05048338e-01, 2.67577176e-01, 3.49173984e-01, 4.55653479e-01,
                            5.94603558e-01, 7.75926020e-01, 1.01254219e+00, 1.32131371e+00,
                            1.72424412e+00])
    N2C_by_NLS_yes = []
    N2C_by_NLS_no = []
    Mechano_by_NLS = []
    for NLS in NLSs_subset:
        N2Cs_no = []
        N2Cs_yes = []
        Mechanos = []
        iNLS = np.argsort(np.abs(NLSs - NLS))[0]
        NLS_match = NLSs[iNLS]
        print(f"\n\n\n----\nNLS match {NLS_match * 1e3} ms^-1")
        is_NLS = df['rate_free_to_complex_per_sec'] == NLS_match
        df_N2C = df[is_NLS].pivot(index='passive_rate',
                                  columns='fraction_complex_NPC_traverse_per_sec',
                                  values='N2C')
        x0 = 15.0
        x1 = 150.0
        ix0 = np.argsort(np.abs(df_N2C.columns - x0))[0]
        ix1 = np.argsort(np.abs(df_N2C.columns - x1))[0]
        for MW in MWs:
            y0 = get_passive_nuclear_molar_rate_per_sec(MW, is_force=False)
            y1 = get_passive_nuclear_molar_rate_per_sec(MW, is_force=True)
            iy0 = np.argsort(np.abs(df_N2C.index - y0))[0]
            iy1 = np.argsort(np.abs(df_N2C.index - y1))[0]
            if (MW > 50):
                iy1 = iy0
            no = df_N2C.iloc[iy0, ix0]
            yes = df_N2C.iloc[iy1, ix1]
            mechano = yes / no
            if VERBOSE:
                print("------")
                print(MW, y0, y1)
                print(f"Mechanosensitivity {mechano:.1f}")
                display(df_N2C.iloc[[iy0, iy1], [ix0, ix1]])
            N2Cs_no.append(no)
            N2Cs_yes.append(yes)
            Mechanos.append(mechano)
            if MW == 41:
                N2C_by_NLS_no.append(no)
                N2C_by_NLS_yes.append(yes)
                Mechano_by_NLS.append(mechano)
        fig, ax = plt.subplots(2, 1, figsize=(3.5, 6), sharex=True)
        ax[0].plot(np.array(MWs) - 1, N2Cs_no, 'bo', label='soft', markersize=4)
        ax[0].plot(np.array(MWs) + 1, N2Cs_yes, 'ro', label='stiff', markersize=4)
        ax[0].set_ylabel("N:C")
        ax[0].legend(frameon=True)
        ax[1].plot(MWs, Mechanos, 'ko')
        ax[1].set_xlabel("MW [kDa]")
        ax[1].set_ylabel("mechanosensitivity")
        ax[1].set_ylim([0.5, 3.0])
        ax[1].set_yscale('log')
        positions = [0.5, 1.0, 2.0, 4.0]
        ax[1].yaxis.set_major_locator(ticker.FixedLocator(positions))
        ax[1].yaxis.set_major_formatter(ticker.FixedFormatter(positions))
        ax[1].yaxis.set_minor_locator(ticker.FixedLocator([]))
        ax[1].yaxis.set_minor_formatter(ticker.FixedFormatter([]))
        ax[0].set_xticks(MWs)
        print("MWs", MWs)
        print("N2C soft", N2Cs_no)
        print("N2C stiff", N2Cs_yes)
        print("mechano:", Mechanos)
        print()

    for xmax in [None, 400.0]:
        fig, ax = plt.subplots(2, 1, figsize=(3.5, 6), sharex=True)
        ax[0].plot(NLSs_subset * 1e3 - 1, N2C_by_NLS_no, 'bo', label='soft', markersize=4)
        ax[0].plot(NLSs_subset * 1e3 + 1, N2C_by_NLS_yes, 'ro', label='stiff', markersize=4)
        ax[0].set_ylabel("N:C")
        ax[0].legend(frameon=True)
        ax[1].plot(NLSs_subset * 1e3, Mechano_by_NLS, 'ko')
        ax[1].set_xlabel(r"NLS strength [$ms^1$]")
        ax[1].set_ylabel("mechanosensitivity")
        ax[1].set_ylim([0.5, 3.0])
        ax[1].set_yscale('log')
        positions = [0.5, 1.0, 2.0, 4.0]
        ax[1].yaxis.set_major_locator(ticker.FixedLocator(positions))
        ax[1].yaxis.set_major_formatter(ticker.FixedFormatter(positions))
        ax[1].yaxis.set_minor_locator(ticker.FixedLocator([]))
        ax[1].yaxis.set_minor_formatter(ticker.FixedFormatter([]))
        ax[0].set_xlim(xmin=1.0, xmax=xmax)
        ax[0].set_xscale('log')
        print("NLSs", NLSs_subset)
        print("N2C soft", N2C_by_NLS_no)
        print("N2C stiff", N2C_by_NLS_yes)
        print("mechano:", Mechano_by_NLS)
        print()
    print("End of cell 51")
    plt.show()


    ## import/export


    #  CELL 53:
    df.head()


    #  CELL 54:
    VERBOSE = False
    PLOT_EACH = False
    plt.style.use('./my.mplstyle.txt')
    NLSs = df['rate_free_to_complex_per_sec'].unique()
    if VERBOSE or True:
        print("NLSs [s^-1]: ", NLSs)
    NLSs_subset = np.array([1.00000000e-03, 1.30494682e-03, 1.70288620e-03, 2.22217592e-03,
                            2.89982140e-03, 3.78411271e-03, 4.93806584e-03, 6.44391330e-03,
                            8.40896415e-03, 1.09732510e-02, 1.43195090e-02, 1.86861977e-02,
                            2.43844942e-02, 3.18204681e-02, 4.15240186e-02, 5.41866359e-02,
                            7.07106781e-02, 9.22736744e-02, 1.20412238e-01, 1.57131566e-01,
                            2.05048338e-01, 2.67577176e-01, 3.49173984e-01, 4.55653479e-01,
                            5.94603558e-01, 7.75926020e-01, 1.01254219e+00, 1.32131371e+00,
                            1.72424412e+00])
    empty_stats = {'import': [],
                   'export': [],
                   'import:export': []}
    Yes_by_NLS = copy.deepcopy(empty_stats)
    No_by_NLS = copy.deepcopy(empty_stats)
    Mechanos_by_NLS = copy.deepcopy(empty_stats)
    for NLS in NLSs_subset:
        No = copy.deepcopy(empty_stats)
        Yes = copy.deepcopy(empty_stats)
        Mechanos = copy.deepcopy(empty_stats)
        iNLS = np.argsort(np.abs(NLSs - NLS))[0]
        NLS_match = NLSs[iNLS]
        print(f"\n\n\n----\nNLS match {NLS_match * 1e3} ms^-1")
        is_NLS = df['rate_free_to_complex_per_sec'] == NLS_match
        df_import = df[is_NLS].pivot(index='passive_rate',
                                     columns='fraction_complex_NPC_traverse_per_sec',
                                     values='nuclear_import_per_sec')
        df_export = df[is_NLS].pivot(index='passive_rate',
                                     columns='fraction_complex_NPC_traverse_per_sec',
                                     values='nuclear_export_per_sec')
        x0 = 15.0
        x1 = 150.0
        ix0 = np.argsort(np.abs(df_N2C.columns - x0))[0]
        ix1 = np.argsort(np.abs(df_N2C.columns - x1))[0]
        for MW in MWs:
            y0 = get_passive_nuclear_molar_rate_per_sec(MW, is_force=False)
            y1 = get_passive_nuclear_molar_rate_per_sec(MW, is_force=True)
            iy0 = np.argsort(np.abs(df_N2C.index - y0))[0]
            iy1 = np.argsort(np.abs(df_N2C.index - y1))[0]
            if (MW > 50):
                iy1 = iy0
            for Stats, ix, iy in zip([No, Yes], [ix0, ix1], [iy0, iy1]):
                Stats['import'].append(df_import.iloc[iy, ix])
                Stats['export'].append(df_export.iloc[iy, ix])
                Stats['import:export'].append(Stats['import'][-1] / Stats['export'][-1])
            for category in empty_stats.keys():
                Mechanos[category].append(Yes[category][-1] / No[category][-1])
            if MW == 41:
                for category in empty_stats.keys():
                    No_by_NLS[category].append(No[category][-1])
                    Yes_by_NLS[category].append(Yes[category][-1])
                    Mechanos_by_NLS[category].append(Mechanos[category][-1])
        if PLOT_EACH:
            fig, ax = plt.subplots(4, 1, figsize=(3.5, 9), sharex=True)
            for i, category in enumerate(['import', 'export', 'import:export']):
                ax[i].plot(np.array(MWs) - 1, No[category], 'bo', label='soft', markersize=4)
                ax[i].plot(np.array(MWs) + 1, Yes[category], 'ro', label='stiff', markersize=4)
                ax[i].set_ylabel(category)
                ax[i].legend(frameon=True)
                ax[i].set_xticks(MWs)
                if category in ['import', 'export']:
                    ax[i].set_yscale('log')
                ax[i].set_ylim(ymax=1.1 * max(max(Yes[category]), max(No[category])))
            ax[-1].plot(MWs, Mechanos[category], 'ko')
            ax[-1].set_xlabel("MW [kDa]")
            ax[-1].set_ylabel("mechanosensitivity")
            ax[-1].set_ylim([0.5, 3.0])
            ax[-1].set_yscale('log')
            positions = [0.5, 1.0, 2.0, 4.0]
            ax[-1].yaxis.set_major_locator(ticker.FixedLocator(positions))
            ax[-1].yaxis.set_major_formatter(ticker.FixedFormatter(positions))
            ax[-1].yaxis.set_minor_locator(ticker.FixedLocator([]))
            ax[-1].yaxis.set_minor_formatter(ticker.FixedFormatter([]))
            print("MWs", MWs)
            print("Soft", No)
            print("Stiff", Yes)
            print("mechano:", Mechanos)
            print()

    for xmax in [None, 400.0]:
        fig, ax = plt.subplots(4, 1, figsize=(3.5, 9), sharex=True)
        for i, category in enumerate(['import', 'export', 'import:export']):
            ax[i].plot(NLSs_subset * 1e3 - 1, No_by_NLS[category], 'bo', label='soft', markersize=4)
            ax[i].plot(NLSs_subset * 1e3 + 1, Yes_by_NLS[category], 'ro', label='stiff', markersize=4)
            ax[i].set_ylabel(category)
            ax[i].legend(frameon=True)
            ax[i].set_xlim(xmin=1.0, xmax=xmax)
            ax[i].set_xscale('log')
            if category in ['import', 'export']:
                ax[i].set_yscale('log')
                ax[i].set_ylim(ymax=1.1 * max(max(Yes[category]), max(No[category])))
        ax[-1].plot(NLSs_subset * 1e3, Mechanos_by_NLS[category], 'ko')
        ax[-1].set_xlabel(r"NLS strength [$ms^1$]")
        ax[-1].set_ylabel("mechanosensitivity")
        ax[-1].set_ylim([0.5, 3.0])
        ax[-1].set_yscale('log')
        positions = [0.5, 1.0, 2.0, 4.0]
        ax[-1].yaxis.set_major_locator(ticker.FixedLocator(positions))
        ax[-1].yaxis.set_major_formatter(ticker.FixedFormatter(positions))
        ax[-1].yaxis.set_minor_locator(ticker.FixedLocator([]))
        ax[-1].yaxis.set_minor_formatter(ticker.FixedFormatter([]))
        print("NLSs", NLSs_subset)
        print("Soft", No_by_NLS)
        print("Stiff", Yes_by_NLS)
        print("Mechano:", Mechano_by_NLS)
        print()
    print("End of cell 54")
    plt.show()


    ## transform


    #  CELL 60:
    v_N_L = 627e-15
    v_C_L = 2194e-15
    N2C_vol = v_N_L / v_C_L

    VERBOSE = False
    plt.style.use('./my.mplstyle.txt')
    NLSs = df['rate_free_to_complex_per_sec'].unique()
    if VERBOSE or True:
        print("NLSs [s^-1]: ", NLSs)
    NLSs_subset = np.array([3.18204681e-02, 4.15240186e-02, 5.41866359e-02,
                            7.07106781e-02, 9.22736744e-02, 1.20412238e-01, 1.57131566e-01,
                            2.05048338e-01, 2.67577176e-01, 3.49173984e-01, 4.55653479e-01,
                            5.94603558e-01])
    Nsat_by_NLS_yes = []
    Nsat_by_NLS_no = []
    Mechano_by_NLS = []
    for NLS in NLSs_subset:
        Nsat_no = []
        Nsat_yes = []
        Mechanos = []
        iNLS = np.argsort(np.abs(NLSs - NLS))[0]
        NLS_match = NLSs[iNLS]
        print(f"\n\n\n----\nNLS match {NLS_match * 1e3} ms^-1")
        is_NLS = df['rate_free_to_complex_per_sec'] == NLS_match
        df_N2C = df[is_NLS].pivot(index='passive_rate',
                                  columns='fraction_complex_NPC_traverse_per_sec',
                                  values='N2C')
        x0 = 15.0
        x1 = 150.0
        ix0 = np.argsort(np.abs(df_N2C.columns - x0))[0]
        ix1 = np.argsort(np.abs(df_N2C.columns - x1))[0]
        for MW in MWs:
            y0 = get_passive_nuclear_molar_rate_per_sec(MW, is_force=False)
            y1 = get_passive_nuclear_molar_rate_per_sec(MW, is_force=True)
            iy0 = np.argsort(np.abs(df_N2C.index - y0))[0]
            iy1 = np.argsort(np.abs(df_N2C.index - y1))[0]
            if (MW > 50):
                iy1 = iy0
            no = df_N2C.iloc[iy0, ix0]
            yes = df_N2C.iloc[iy1, ix1]
            mechano = yes / no
            if VERBOSE:
                print("------")
                print(MW, y0, y1)
                print(f"Mechanosensitivity {mechano:.1f}")
                display(df_N2C.iloc[[iy0, iy1], [ix0, ix1]])
            Nsat_no.append(transform_N2C_to_N_relative(no, N2C_vol))
            Nsat_yes.append(transform_N2C_to_N_relative(yes, N2C_vol))
            Mechanos.append(mechano)
            if MW == 41:
                Nsat_by_NLS_no.append(transform_N2C_to_N_relative(no, N2C_vol))
                Nsat_by_NLS_yes.append(transform_N2C_to_N_relative(yes, N2C_vol))
                Mechano_by_NLS.append(mechano)
        fig, ax = plt.subplots(2, 1, figsize=(3.5, 6), sharex=True)
        ax[0].plot(np.array(MWs) - 1, Nsat_no,
                   'bo', label='soft', markersize=4)
        ax[0].plot(np.array(MWs) + 1, Nsat_yes,
                   'ro', label='stiff', markersize=4)
        ax[0].set_xticks(MWs)
        ax[0].set_ylabel("N saturation")
        ax[0].legend(frameon=True)
        ax[0].set_ylim(0.1, 1)
        ax[0].set_yscale('log')
        ax[1].plot(MWs, Mechanos, 'ko')
        ax[1].set_xlabel("MW [kDa]")
        ax[1].set_ylabel("mechanosensitivity")
        ax[1].set_ylim([0.5, 3.0])
        ax[1].set_yscale('log')
        positions = [0.5, 1.0, 2.0, 4.0]
        ax[1].yaxis.set_major_locator(ticker.FixedLocator(positions))
        ax[1].yaxis.set_major_formatter(ticker.FixedFormatter(positions))
        ax[1].yaxis.set_minor_locator(ticker.FixedLocator([]))
        ax[1].yaxis.set_minor_formatter(ticker.FixedFormatter([]))
        print("MWs", MWs)
        print("N sat. soft", Nsat_no)
        print("N sat. stiff", Nsat_yes)
        print("mechano:", Mechanos)
        print()

    fig, ax = plt.subplots(2, 1, figsize=(3.5, 6), sharex=True)
    ax[0].plot(NLSs_subset * 1e3 - 1, Nsat_by_NLS_no, 'bo', label='soft', markersize=4)
    ax[0].plot(NLSs_subset * 1e3 + 1, Nsat_by_NLS_yes, 'ro', label='stiff', markersize=4)
    ax[0].set_ylabel("N saturation")
    ax[0].set_ylim(0.1, 1)
    ax[0].set_yscale('log')
    ax[0].legend(frameon=True)
    ax[1].plot(NLSs_subset * 1e3, Mechano_by_NLS, 'ko')
    ax[1].set_xlabel(r"NLS strength [$ms^1$]")
    ax[1].set_ylabel("mechanosensitivity")
    ax[1].set_ylim([0.5, 3.0])
    ax[1].set_yscale('log')
    positions = [0.5, 1.0, 2.0, 4.0]
    ax[1].yaxis.set_major_locator(ticker.FixedLocator(positions))
    ax[1].yaxis.set_major_formatter(ticker.FixedFormatter(positions))
    ax[1].yaxis.set_minor_locator(ticker.FixedLocator([]))
    ax[1].yaxis.set_minor_formatter(ticker.FixedFormatter([]))
    print("NLSs", NLSs_subset)
    print("Nsat soft", Nsat_by_NLS_no)
    print("Nsat stiff", Nsat_by_NLS_yes)
    print("mechano:", Mechano_by_NLS)
    print()
    print("End of cell 60")
    plt.show()


    ## Normalized dX, dY and Z (transformed or not)


    #  CELL 62:
    plt.style.use('./my.mplstyle.txt')
    v_N_L = 627e-15
    v_C_L = 2194e-15
    N2C_vol = v_N_L / v_C_L
    print(f"{N2C_vol:.2f}")
    for is_transform in [False, True]:
        print(f"is_transform {is_transform}")
        plot_dX_dY_Z(df,
                     is_transform=is_transform,
                     N2C_vol=N2C_vol)
    print("End of cell 62")
    plt.show()


    #  CELL 63:
    plt.style.use('./my.mplstyle.txt')
    traverses = df['fraction_complex_NPC_traverse_per_sec'].unique()[:]
    ncol = 3
    nrow = len(traverses) // ncol + (len(traverses) % ncol > 0) * 1
    fig, axes = plt.subplots(nrow, ncol,
                             sharex=True,
                             sharey=True,
                             squeeze=False,
                             figsize=(17.0, 4.0 * nrow + 1.0))
    axes_iter = iter(axes.reshape(-1))
    for traverse in traverses:
        ax = next(axes_iter)
        is_traverse = np.isclose(df['fraction_complex_NPC_traverse_per_sec'], traverse)
        N2C = df[is_traverse].pivot(index='rate_free_to_complex_per_sec',
                                    columns='passive_rate',
                                    values='N2C')
        NC_min = 1.0
        NC_max = 10.0
        plt.sca(ax)
        ct = my_plot_param_grid(N2C,
                                pretty_x=r'passive rate [$sec^{-1}$]',
                                pretty_y=r'NLS $k_{on}$ [$sec^{-1}$]',
                                pretty_z=r'N:C',
                                vmin=NC_min,
                                vmax=NC_max,
                                levels=np.logspace(np.log2(NC_min), np.log2(NC_max), 21, base=2.0),
                                locator=mpl.ticker.LogLocator(base=2.0),
                                extend='both')
        colors = ['r', 'g', 'b', 'k']
        xlim = ax.get_xlim()
        for MW, color in zip(MWs, colors):
            x0 = get_passive_nuclear_molar_rate_per_sec(MW, is_force=False)
            x1 = get_passive_nuclear_molar_rate_per_sec(MW, is_force=True)
            x1 = min(x1, xlim[1] * 0.99)
            y0 = N2C.index[0]
            y1 = N2C.index[-1]
            ax.plot((x0, x0), (y0, y1), color + "-", label=str(MW))
            plt.annotate(str(MW), (x0, y0), color=color)
            ax.plot((x1, x1), (y0, y1), color + "-", linewidth=3.0)
        ax.set_xlim(xlim)
        plt.title(f"rate NPC traverse ${traverse:.1f} $sec^{{-1}}$")
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.3, 0.03, 0.4])
    cb = fig.colorbar(ct, cax=cbar_ax)
    ticks = cb.get_ticks()
    cb.set_ticks(ticks)
    cb.set_ticklabels(["{:.2f}".format(tick) for tick in ticks])
    print("End of cell 63")
    plt.show()


    #  CELL 64:
    plot_cells()
    print("End of cell 64")
    plt.show()

    ## next one

    #  CELL 66:
    dRatios_by_passive = get_dLogRatios_by_passive(stats_grids_by_passive, ts_by_passive)
    for key in sorted(stats_grids_by_passive.keys())[:-1]:
        print(key)
        map_param_grid.plot_param_grid(param_range, dRatios_by_passive[key], Z_label="dN/C log ratio", vmin=-4.0, vmax=1.0,
                            levels=np.linspace(-4.0, 1.0, 21), extend='both')
    print("End of cell 66")
    plt.show()


    #  CELL 67:
    """
    keys = sorted(stats_grids_traverse_by_passive_force.keys())
    for key in keys:
        print(f"passive rate {key}")
        plot_stats_grids(stats_grids_traverse_by_passive_force[key], ts_traverse_by_passive_force[key],
                         vmax_import_export=10.0, NC_max=30.0, NC_min=1.0)

    # Map NLS strength, MW size, force
    """


    #  CELL 69:
    importlib.reload(transport_simulation)


    free_to_complex_rates = [
        #0.0,
        #0.001,
        #0.00316,
        #0.01,
        #0.02 #2.11
        0.045,  #2.11
        0.1,  #16.4
        0.2
        #0.45,
        #1.0,
        #2.0,
        #4.5
    ]


    #  CELL 73:
    ##### TIME CONSUMING #####$
    simulation_time_sec = 40.0
    n_processors = os.cpu_count()
    MW_to_stats_list_by_force = {}
    for MW in [27, 41, 54, 67]:
        print(MW)
        MW_to_stats_list_by_force[MW] = get_MW_stats_list_by_force(MW, simulation_time_sec, n_processors=n_processors,
                                                                   is_change_cell_volume=False)
        plot_MW_stats_list(*MW_to_stats_list_by_force[MW])
    print("End of cell 73")
    plt.show()
    time.sleep(5)


    #  CELL 77:
    ##### TIME CONSUMING #####$
    simulation_time_sec = 40.0
    n_processors = os.cpu_count()
    MW_to_stats_list_by_force = {}
    for MW in [27, 41, 54, 67]:
        print(MW)
        MW_to_stats_list_by_force[MW] = get_MW_stats_list_by_force(MW, simulation_time_sec, n_processors=n_processors,
                                                                   is_change_cell_volume=True)
        plot_MW_stats_list(*MW_to_stats_list_by_force[MW])
    print("End of cell 77")
    plt.show()
    time.sleep(5)


    #  CELL 79:
    with open("MW_to_stats_list_by_force.pkl", "wb") as F:
        pickle.dump(MW_to_stats_list_by_force, F)
