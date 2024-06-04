# ---
# jupyter:
#   jupytext:
#     formats: py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import multiprocess
import pdb
import matplotlib.axes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import transport_simulation
import pandas as pd

# %%
N_A = 6.022e+23  # Avogadro's number


def get_tau_passive_diffusion(nmol_per_sec_per_M: float, volume_L: float) -> float:
    """
    This utility function computes tau for one-sided passive diffusion
    specified in number of molecules per second per M, given
    the volume from which the passive diffusion leaves

    :param nmol_per_sec_per_M: passive diffusion rate (in units number of molecules/sec/M)
    :param volume_L: initial volume for diffusion (in liters)
    :return: characteristic time (in sec)
    TODO: is this needed anywhere?
    TODO: need to check this one
    """
    global N_A
    gamma = nmol_per_sec_per_M / N_A / volume_L
    tau = 1.0 / gamma
    return tau


def get_new_transport_simulation(**kwargs) -> transport_simulation.TransportSimulation:
    """
    Generates a new transport simulation given a set of params

    :return: a new transport simulation instance
    """
    ts = transport_simulation.TransportSimulation(**kwargs)
    ts.set_time_step(1e-3)
    return ts


def mp_do_simulation(param_range: dict, i: int, j: int,
                     equilibration_time_sec: float,
                     tsg,  # transport simulation generator
                     tsg_params: dict,  # transport_simulation_generator_params
                     rng: np.random.Generator  # random number generator
                     ) -> dict:
    """
    Runs the transport simulation for a given x-y pairing (given by values i and j) and system equilibration time and
    returns the results

    :param param_range: a dictionary with 'tag_x'/'tag_y' keys corresponding
                        to x/y-axis param names, resp., and 'range_x'/'range_y' keys
                        for numpy array for corresponding param ranges
    :param i: index for the x value in the x-range to simulate on
    :param j: index for the y value in the y-range to simulate on
    :param equilibration_time_sec: time for system to equilibrate (in seconds)
    :param tsg: transport simulator generator function
    :param tsg_params: dictionary of param names and values to be passed to the transport simulator generator function
    :param rng: a random number generator
    :return: dictionary with results from the simulation, as well as the i and j indexes for the simulation
    """
    print("\nStarting simulation for combo: i-{}, j-{}\n".format(i, j))
    nskip_statistics = 100
    my_ts = tsg(**tsg_params)
    cur_params = {param_range['tag_x']: param_range['range_x'][i],
                  param_range['tag_y']: param_range['range_y'][j]}
    my_ts.set_params(**cur_params)
    stats = my_ts.simulate(equilibration_time_sec,
                           nskip_statistics=100)
    r = rng.random()
    if r < 0.1:
        print(f"Finished some ten jobs (at i={i} j={j} r={r})")
    print("\nFinished simulation for combo: i-{}, j-{}\n".format(i, j))
    return {"i": i, "j": j, "stats": stats}


def mp_handle_stats(stats_grids: dict, mydicts: list) -> None:
    """
    Update the master dictionary of species values for each x-y value pairing with the results of their respective
    simulations

    :param stats_grids: dictionary containing species information at the end of each simulation in the form of numpy
                        arrays (where each j-i pair is the result of a particular y-x pairing)
    :param mydicts: list of dictionaries for each i-j pair that contain the values for each species at each time point
                    frame
    :return: None
    """
    for mydict in mydicts:
        i = mydict["i"]
        j = mydict["j"]
        stats = mydict["stats"]
        #        print("My Handle - ", i, j)
        for nmol_type in stats.keys():
            if nmol_type == "time_sec":
                continue
            try:
                pass
                stats_grids[nmol_type][j, i] = stats[nmol_type][-1]  # matrix is indexed row first
            except KeyError:
                print(f"my_handle_stats - Key {nmol_type} not found")
            except IndexError as e:
                print(e)


def mp_handle_error(error: Exception) -> None:
    """
    Prints out the error that is passed in

    :param error: the error that needs to be printed
    :return: None
    """
    import traceback
    print("Error", error)
    print("Traceback: \n", traceback.format_exc())


def map_param_grid_parallel(param_range: dict,
                            equilibration_time_sec: float = 150.0,
                            n_processors: int = 5,
                            transport_simulation_generator=get_new_transport_simulation,
                            transport_simulation_generator_params=dict()
                            ) -> tuple:
    """
    Runs multiple simulations in parallel over a range of parameters (provided by param_range)

    :param param_range: a dictionary with 'tag_x'/'tag_y' keys corresponding
                        to x/y-axis param names, resp., and 'range_x'/'range_y' keys
                        for numpy array for corresponding param ranges
    :param equilibration_time_sec: system equilibration time in seconds (regardless of condition)
    :param n_processors: number of processors on which to run in parallel
    :param transport_simulation_generator: a picklable (global) function that generates
                                           transport_simulation objects
    :param transport_simulation_generator_params: dictionary of parameters to pass to the
                                                  transport_simulation_generator
    :return: a tuple containing a dictionary of simulation properties with their values at the end of simulations for
             each parameter combination (stored as a numpy array indexed y-x) and a sample transport simulation to
             obtain some basic information (e.g. cytoplasmic and nuclear volumes)
    """
    VERBOSE = True
    nx = len(param_range['range_x'])
    ny = len(param_range['range_y'])
    ts = transport_simulation_generator(**transport_simulation_generator_params)  # a sample ts to be returned
    if VERBOSE:
        for param in [param_range['tag_x'], param_range['tag_y']]:
            print("Param {} default value is {}".format(param, getattr(ts, param)))
    stats_grids = {}
    for nmol_type in ts.nmol.keys():  # TODO: add get_nmols()
        stats_grids[nmol_type] = np.ndarray((ny, nx))  # Row major - y coordinate goes first
    jobs_params = []
    seeds = np.random.SeedSequence().spawn(nx * ny)
    rngs = [np.random.default_rng(s) for s in seeds]
    for j in range(ny):
        for i in range(nx):
            jobs_params.append((param_range.copy(),
                                i, j,
                                equilibration_time_sec,
                                transport_simulation_generator,
                                transport_simulation_generator_params,
                                rngs[j * nx + i]))

    print("njobs={}".format(len(jobs_params)))
    callback_function = lambda mydict: mp_handle_stats(stats_grids, mydict)
    pool = multiprocess.Pool(processes=n_processors)
    results = pool.starmap_async(mp_do_simulation,
                                 jobs_params,
                                 callback=callback_function,
                                 error_callback=mp_handle_error)
    results.wait()
    pool.close()
    pool.join()
    return stats_grids, ts


def _get_df_for_stats_grids(param_range: dict, passive_rate, stats_grids: dict,
                            ts: transport_simulation.TransportSimulation) -> pd.DataFrame:
    """

    :param param_range: a dictionary with 'tag_x'/'tag_y' keys corresponding
                        to x/y-axis param names, resp., and 'range_x'/'range_y' keys
                        for numpy array for corresponding param ranges
    :param passive_rate: the rate of passive diffusion in transport
       simulations
    :param stats_grids: dictionary containing species information at the end of each simulation in the form of numpy
                        arrays (where each j-i pair is the result of a particular y-x pairing)
    :param ts: an instance of TransportSimulation (to obtain cytoplasmic and nuclear volumes)
    :return:
    """
    # TODO: update docstring and typing
    pdb.set_trace()
    X, Y = np.meshgrid(param_range['range_x'],
                       param_range['range_y'])
    df = pd.DataFrame(data={param_range['tag_x']: X.flatten(),
                            param_range['tag_y']: Y.flatten()})
    df['passive_rate'] = passive_rate
    for Z_column, Z_values in stats_grids.items():
        df[Z_column] = Z_values.flatten()
        # Add various useful columns:
    df['V_N_L'] = ts.get_v_N_L()
    df['V_C_L'] = ts.get_v_C_L()
    df['cargo_N_M'] = \
        sum([df[f'{s}{l}_N'] for s in ['free', 'complex'] for l in ['L', 'U']]) / df['V_N_L'] / N_A
    df['cargo_C_M'] = \
        sum([df[f'{s}{l}_C'] for s in ['free', 'complex'] for l in ['L', 'U']]) / df['V_C_L'] / N_A
    df['N2C'] = df['cargo_N_M'] / (df['cargo_C_M'] + 1e-18)
    for d in ['import', 'export']:
        df[f'nuclear_{d}_per_sec'] = \
            sum(df[f'nuclear_{d}{tag}_per_sec'] for tag in ['L', 'U'])
    df['import2export'] = df['nuclear_import_per_sec'] / (df['nuclear_export_per_sec'] + 1e-18)

    return df


def get_df_from_stats_grids_by_passive(param_range: dict,
                                       stats_grids_by_passive,
                                       ts_by_passive) -> pd.DataFrame:
    """
    Converts stats_grids_by_passive returned by map_param_grid_parallel() etc.
    to a pandas dataframe with column 'passive rate' for passive rates, and
    x-range and y-range params fetched from param_range

    :param param_range: a dictionary with 'tag_x'/'tag_y' keys corresponding
                        to x/y-axis param names, resp., and 'range_x'/'range_y' keys
                        for numpy array for corresponding param ranges
    :param stats_grids: dictionary containing species information at the end of each simulation in the form of numpy
                        arrays (where each j-i pair is the result of a particular y-x pairing)
    :param stats_grids_by_passive:
    :param ts_by_passive:
    :return:
    """
    # TODO: update docstring and typing
    pdb.set_trace()
    dfs = [_get_df_for_stats_grids(param_range,
                                   passive,
                                   stats_grids,
                                   ts_by_passive[passive])
           for (passive, stats_grids) in stats_grids_by_passive.items()]
    df = pd.concat(dfs)
    return df


###### VISUALIZATION OF RESULTS ######

def plot_param_grid(param_range: dict,
                    Z: np.ndarray,
                    Z_label: str = None,
                    **contourf_kwargs) -> None:
    """
    Creates a contour plot given a set of x, y, and z param names and values

    :param param_range: a dictionary with 'tag_x'/'tag_y' keys corresponding
                        to x/y-axis param names, resp., and 'range_x'/'range_y' keys
                        for numpy array for corresponding param ranges
    :param Z: the z values to be plotted on the contour plot
    :param Z_label: the label for the z measurement of the contour plot
    :param contourf_kwargs: kwargs that will be passed to the matplotlib contourf function
    :return: None
    """
    try:
        x_meshgrid, y_meshgrid = np.meshgrid(param_range["range_x"],
                                             param_range["range_y"])
        plt.contourf(x_meshgrid,
                     y_meshgrid,
                     Z,
                     **contourf_kwargs)
        #    print("contourf params", contourf_kwargs)
        ax = plt.gca()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(param_range['pretty_x'])
        ax.set_ylabel(param_range['pretty_y'])
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if xlim[1] > xlim[0]:
            ax.set_xlim(xlim[1], xlim[0])
        cb = plt.colorbar(label=Z_label)
        ticks = cb.get_ticks()
        cb.set_ticks(ticks)
        cb.set_ticklabels(["{:.2f}".format(tick) for tick in ticks])
    except ValueError as e:
        if str(e) == "upper_level must be larger than lower_level":
            print("\nUpper level/lower level error:\n"+ str(e) + '\n')
        else:
            raise e


def get_N_to_C_ratios(stats_grids: dict, v_N_L: float, v_C_L: float) -> np.ndarray:
    """
    Computes the cargo nuclear-to-cytoplasmic ratios (normalized by area volumes) for the simulation of each x-y pairing

    :param stats_grids: dictionary containing species information at the end of each simulation in the form of numpy
                        arrays (where each j-i pair is the result of a particular y-x pairing)
    :param v_N_L: volume (in liters) of the nucleus
    :param v_C_L: volume (in liters) of the cytoplasm
    :return: numpy array with the N/C ratios where each i-j pairing is the same as in stats_grids
    """
    nNs = stats_grids["complexL_N"] + stats_grids["freeL_N"] + stats_grids["complexU_N"] + stats_grids["freeU_N"]
    nCs = stats_grids["complexL_C"] + stats_grids["freeL_C"] + stats_grids["complexU_C"] + stats_grids["freeU_C"]
    ratios = (nNs / v_N_L) / (nCs / v_C_L)
    return ratios


def get_N_to_C_ratios_for_ts(stats_grids: dict, ts: transport_simulation.TransportSimulation) -> np.ndarray:
    """
    Computes the cargo nuclear-to-cytoplasmic ratios (normalized by area volumes) for the simulation of each x-y pairing

    :param stats_grids: dictionary containing species information at the end of each simulation in the form of numpy
                        arrays (where each j-i pair is the result of a particular y-x pairing)
    :param ts: transport simulation instance used to obtain nuclear and cytoplasmic volumes
    :return: numpy array with the N/C ratios where each i-j pairing is the same as in stats_grids
    """
    return get_N_to_C_ratios(stats_grids,
                             v_N_L=ts.get_v_N_L(),
                             v_C_L=ts.get_v_C_L())


def plot_NC_ratios(param_range: dict, stats_grids: dict, ts: transport_simulation.TransportSimulation,
                   vmin: float = 1.0, vmax:float = 4.0,
                   locator: mtick.LogLocator = None, levels: np.ndarray = None) -> None:
    """
    Creates a contour plot with N/C ratios as the z-values

    :param param_range: a dictionary with 'tag_x'/'tag_y' keys corresponding to x/y-axis param names, resp., and
                        'range_x'/'range_y' keys for numpy array for corresponding param ranges
    :param stats_grids: dictionary containing species information at the end of each simulation in the form of numpy
                        arrays (where each j-i pair is the result of a particular y-x pairing)
    :param ts: transport simulation instance used to obtain nuclear and cytoplasmic volumes
    :param vmin: min value for the contour plot color range
    :param vmax: max value for the contour plot color range
    :param locator: used to determine levels for the contour plot in case levels are not given
    :param levels: determines values for the contour lines and regions (must be in increasing order)
    :return: None
    """
    NC_ratios = get_N_to_C_ratios_for_ts(stats_grids, ts)
    print(vmin, vmax)
    plot_param_grid(param_range,
                    NC_ratios,
                    Z_label="N/C ratio",
                    vmin=vmin,
                    vmax=vmax,
                    locator=locator,
                    levels=levels,
                    extend='both')


def plot_bound_fraction(param_range: dict, stats_grids: dict, compartment: str,
                        ax: matplotlib.axes.Axes = None) -> None:
    """
    Computes the fraction of complexed labeled cargo in a particular area for the simulation of each x-y pairing
    and creates a contour plot

    :param param_range: a dictionary with 'tag_x'/'tag_y' keys corresponding
                        to x/y-axis param names, resp., and 'range_x'/'range_y' keys
                        for numpy array for corresponding param ranges
    :param stats_grids: dictionary containing species information at the end of each simulation in the form of numpy
                        arrays (where each j-i pair is the result of a particular y-x pairing)
    :param compartment: area for which to calculate the bound fraction (nucleus or cytoplasm)
    :param ax: The axes to be set for the contour plot
    :return: None
    """
    assert (compartment in ['N', 'C'])
    tag_complex = f"complexL_{compartment}"
    tag_free = f"freeL_{compartment}"
    complexL_fraction = stats_grids[tag_complex] / (stats_grids[tag_complex] + stats_grids[tag_free])
    # Contourf
    vmax = 1.0
    if ax is not None:
        plt.sca(ax)
    print(param_range)
    plot_param_grid(param_range,
                    complexL_fraction,
                    Z_label=f"bound fraction ({compartment})",
                    vmin=0.0,
                    vmax=vmax,
                    levels=np.linspace(0, vmax, 11),
                    extend='both')


def get_import_export_ratios(stats_grids: dict) -> np.ndarray:
    """
    Calculates the cargo nuclear-import-to-nuclear-export ratios for the simulation of each x-y pairing

    :param stats_grids: dictionary containing species information at the end of each simulation in the form of numpy
                        arrays (where each j-i pair is the result of a particular y-x pairing)
    :return: numpy array with the import/export ratios where each i-j pairing is the same as in stats_grids
    """
    import_rate = stats_grids["nuclear_importL_per_sec"] + stats_grids["nuclear_importU_per_sec"]
    export_rate = stats_grids["nuclear_exportL_per_sec"] + stats_grids["nuclear_exportU_per_sec"]
    ratios = import_rate / export_rate
    return ratios


def plot_import_export(param_range: dict, stats_grids: dict, axes: list = None,
                       **contourf_kwargs) -> None:
    """
    Creates contour plots of the import and export rates of labeled cargo in and out of the nucleus

    :param param_range: a dictionary with 'tag_x'/'tag_y' keys corresponding
                        to x/y-axis param names, resp., and 'range_x'/'range_y' keys
                        for numpy array for corresponding param ranges
    :param stats_grids: dictionary containing species information at the end of each simulation in the form of numpy
                        arrays (where each j-i pair is the result of a particular y-x pairing)
    :param axes: axes to use for the different contour plots
    :param contourf_kwargs: kwargs that will be passed to the matplotlib contourf function
    :return: None
    """
    if axes is None:
        fig, axes = subplot(2, 1)
    for tag, ax in zip(['import', 'export'], axes):
        plt.sca(ax)
        plot_param_grid(param_range,
                        stats_grids[f'nuclear_{tag}L_per_sec'],
                        Z_label=tag + r' [$sec^{-1}$]',
                        **contourf_kwargs)
