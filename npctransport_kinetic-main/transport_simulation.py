import numpy as np
import re

N_A = 6.022e+23  # Avogadro's number
VERBOSE = True
CHECKS = False


# TODO: test
def fL_to_L(v_fL: float) -> float:
    """
    Converts femtoliters to liters

    :param v_fL: volume in femtoliters
    :return: liter value corresponded to given femtoliter value
    """
    return v_fL * 1E-15


def register_update_functions(cls):
    """
    A class decorator that allows registering methods. 
    Methods registered will be added to a class variable list called '_update_funcs'
    To register a function, use the decorator 'register_update()' defined below.

    :param cls: the desired class to be decorated
    """
    cls._update_funcs = []
    for methodname in dir(cls):
        method = getattr(cls, methodname)
        if hasattr(method, '_update_func'):
            if method._update_func:
                cls._update_funcs.append(method)
                if VERBOSE:
                    print(f"Registering {method}")
    return cls


def register_update(active: bool = True):
    """
    A function which created a decorator that adds a function a bool attribute called '_update_func'
    with the value 'active' This is used to tag functions that can update the species transition list during the
    simulation

    :param active: whether the function is a function that updates the transition list
    """
    def wrapper(func):
        """
        Wrapper function that adds an _update_func attribute to the function that is passed in
        """
        func._update_func = active
        return func

    return wrapper


def register_move_nmol(T_list: list, src: str, dst: str, nmol: float) -> None:
    """
    Appends a species transition to the species transition list

    :param T_list: a list of tuples (src, dst, nmol), each representing a transition of a number of molecules (nmol)
                       from a molecular species source (src) to a molecular species destination (dst)
    :param src: the source species
    :param dst: the destination species
    :param nmol: the number of molecules
    :return: None
    """
    T_list.append((src, dst, nmol))


@register_update_functions
class TransportSimulation:

    ###########################################
    # getter/setter functions / utility
    ###########################################

    def set_nmol(self, species: str, value: float) -> None:
        """
        Saves the molecule value for a given species

        :param species: species tag
        :param value: number of molecules of the given species
        :return: None
        """
        self.nmol[species] = value

    def get_nmol(self, species: str) -> float:
        """
        Returns the number of molecules of a given species in its compartment

        :param species: species tag
        """
        return self.nmol[species]

    def get_compartment(self, species: str) -> any:
        """
        Determines the compartment associated with a given species tag

        :param species: species tag
        :return: string containing N,C, or NPC, None if species tag is irregular
        """
        m = re.search("(NPC_[A-Z]*)_..port$", species)
        if m is not None:
            return "NPC"
        m = re.search("[a-zA-Z*]_([A-Z]*)$", species)
        if m is None:
            return None
        assert (m.group(1) in ['N', 'C'])
        return m.group(1)

    def get_compartment_volume_L(self, species: str) -> float:
        """
        Returns the volume of the comportment for a species given the species tag

        :param species: species tag
        :return: volume (in liters) of the species compartment
        """
        compt = self.get_compartment(species)
        if compt == "N":
            return self.v_N_L
        elif compt == "C":
            return self.v_C_L
        elif compt == "cell":
            return self.v_N_L + self.v_C_L
        else:
            raise ValueError(f"Only nucleus/cytoplasm/cell has a volume (species {species} compartment {compt})")

    def set_concentration_M(self, species: str, c_M: float) -> None:
        """
        Stores the number of molecules of specified species given its concentration

        :param species: species tag
        :param c_M: concentration (in molar) of the species in its compartment
        :return: None
        :raise ValueError: if compartment has no volume
        """
        global N_A
        v_L = self.get_compartment_volume_L(species)
        if v_L == 0:
            raise ValueError(f"Volume for compartment of {species} is 0!")
        self.nmol[species] = c_M * v_L * N_A

    def get_concentration_M(self, species: str) -> float:
        """
        Calculates the concentration of specified species in its compartment

        :param species: species tag
        :return: concentration (in molar) of the given species in its compartment
        :raise ValueError: if compartment has no volume
        """
        global N_A
        v_L = self.get_compartment_volume_L(species)
        if v_L == 0:
            raise ValueError(f"Volume for compartment of {species} is 0!")
        return self.nmol[species] / (N_A * v_L)

    def set_RAN_distribution(self, Ran_cell_M: float, parts_GTP_N: float, parts_GTP_C: float, parts_GDP_N: float,
                             parts_GDP_C: float) -> None:
        """
        Sets the RAN distribution among compartments based on relative parts specified, s.t. total number of RAN
        molecules is constant

        :param Ran_cell_M: total Ran concentration in the nucleus and cytoplasm combined (in M)
        :param parts_GTP_N: number of GTP parts in the nucleus
        :param parts_GTP_C: number of GTP parts in the cytoplasm
        :param parts_GDP_N: number of GDP parts in the nucleus
        :param parts_GDP_C: number of GDP parts in the cytoplasm
        """
        global N_A
        RAN_distribution = np.array([parts_GTP_N, parts_GTP_C,
                                     parts_GDP_N, parts_GDP_C])
        RAN_distribution = RAN_distribution / np.sum(RAN_distribution)  # normalize to 1
        nmol_Ran_cell = Ran_cell_M * (self.v_N_L + self.v_C_L) * N_A
        self.nmol["GTP_N"] = nmol_Ran_cell * RAN_distribution[0]
        self.nmol["GTP_C"] = nmol_Ran_cell * RAN_distribution[1]
        self.nmol["GDP_N"] = nmol_Ran_cell * RAN_distribution[2]
        self.nmol["GDP_C"] = nmol_Ran_cell * RAN_distribution[3]

    def reset_cargo_concentration(self, cargo_cytoplasmic_M: float, fraction_bound: float = 0.0) -> None:
        """
        Resets all cargo concentrations to 0 and sets the cytoplasmic cargo concentrations based on params provided

        :param cargo_cytoplasmic_M: total concentration (in molar) of labeled cargo (bounded and unbound) in the
                                    cytoplasm
        :param fraction_bound: fraction of labeled cargo bound to importin to form cargo-importin complex
        :return: None
        """
        # Cytoplasm:
        self.set_concentration_M("complexL_C",
                                 fraction_bound * cargo_cytoplasmic_M)  # Cytoplasmic concentration of labeled cargo-importin complex in M
        self.set_concentration_M("freeL_C",
                                 (
                                         1.0 - fraction_bound) * cargo_cytoplasmic_M)  # Cytoplasmic concentration of labeled cargo in M
        self.nmol["complexU_C"] = 0  # (unlabeled)
        self.nmol["freeU_C"] = 0  # (unlabeled)
        # Nucleus:
        self.set_concentration_M("cargo_N", 0e-5)  # Nuclear concentration of labeled cargo in M
        self.nmol["complexL_N"] = 0  # number of cargo-importin complexes in nucleus (labeled)
        self.nmol["freeL_N"] = self.nmol["cargo_N"] - self.nmol[
            "complexL_N"]  # number of free cargo molecules in nucleus (labeled)
        self.nmol["complexU_N"] = 0  # (unlabeled)
        self.nmol["freeU_N"] = 0  # (unlabeled)
        del self.nmol["cargo_N"]
        # NPC:
        self.nmol[
            "complexL_NPC_N_import"] = 0  # number of cargo-importin complexes docked to the NPC on the nucleus side (labeled)
        self.nmol[
            "complexL_NPC_C_import"] = 0  # number of cargo-importin complexes docked to the NPC on the cytoplasmic side (labeled)
        self.nmol[
            "complexU_NPC_N_import"] = 0  # number of cargo-importin complexes docked to the NPC on the nucleus side (unlabeled)
        self.nmol[
            "complexU_NPC_C_import"] = 0  # number of cargo-importin complexes docked to the NPC on the cytoplasmic side (unlabeled)
        self.nmol[
            "complexL_NPC_N_export"] = 0  # number of cargo-importin complexes docked to the NPC on the nucleus side (labeled)
        self.nmol[
            "complexL_NPC_C_export"] = 0  # number of cargo-importin complexes docked to the NPC on the cytoplasmic side (labeled)
        self.nmol[
            "complexU_NPC_N_export"] = 0  # number of cargo-importin complexes docked to the NPC on the nucleus side (unlabeled)
        self.nmol[
            "complexU_NPC_C_export"] = 0  # number of cargo-importin complexes docked to the NPC on the cytoplasmic side (unlabeled)

    def set_v_N_L(self, v_L: float, fix_concentration: bool) -> None:
        """
        Changes nuclear volume

        :param v_L: new volume in liters
        :param fix_concentration: whether to rescale number of nuclear molecules to fix nuclear (not cellular)
                                  concentration
        :return: None
        """
        print(f"change v_N_L from {self.v_N_L} to {v_L}")
        if fix_concentration:
            s = v_L / self.v_N_L
            for key, value in self.nmol.items():
                if self.get_compartment(key) == 'N':
                    print(f"Scaling {key} by {s} from {value}")
                    self.set_nmol(key, s * value)
        self.v_N_L = v_L

    def set_v_C_L(self, v_L: float, fix_concentration: bool) -> None:
        """
        Changes cytoplasmic volume

        :param v_L: new volume in liters
        :param fix_concentration: whether to rescale number of cytoplasmic molecules to fix cytoplasmic (not cellular)
                                  concentration
        :return: None
        """
        print(f"change v_C_L from {self.v_C_L} to {v_L}")
        if fix_concentration:
            s = v_L / self.v_C_L
            for key, value in self.nmol.items():
                if self.get_compartment(key) == 'C':
                    print(f"Scaling {key} by {s} from {value}")
                    self.set_nmol(key, s * value)
        self.v_C_L = v_L

    def get_v_N_L(self) -> float:
        """
        Returns the volume of the nucleus in liters
        """
        return self.v_N_L

    def get_v_C_L(self) -> float:
        """
        Returns the volume of the cytoplasm in liters
        """
        return self.v_C_L

    def get_v_cell_L(self) -> float:
        """
        Returns the volume of the cell in liters (adds vol of nucleus and vol of cytoplasm)
        """
        return self.v_N_L + self.v_C_L

    def set_time_step(self, dt_sec: float) -> None:
        """
        Sets time step for simulation

        :param dt_sec: the time step (in seconds) for each interation of the simulation
        :return: None
        """
        self.dt_sec = dt_sec

    def get_time_step(self) -> float:
        """
        Returns time step in seconds
        """
        return self.dt_sec

    def reset_simulation_time(self) -> None:
        """
        Resets simulation time value back to 0

        :return: None
        """
        self.sim_time_sec = 0.0

    ###################
    # Constructor (and init functions)
    ###################
    def set_passive_nuclear_molar_rate_per_sec(self, rate_per_sec: float) -> None:
        """
        Sets the parameter for maximum passive rate of nuclear diffusion such that the passive component of
        d[N]/dt is rate_per_sec*([C]-[N])

        :param rate_per_sec: the max rate of passive diffusion in 1/(second*M)
        :return: None
        """
        self.max_passive_diffusion_rate_nmol_per_sec_per_M = \
            rate_per_sec * N_A * self.v_N_L  # convert per_M to per_nmol

    def set_passive_cytoplasmic_molar_rate_per_sec(self, rate_per_sec: float) -> None:
        """
        Sets the parameter for maximum passive rate of cytoplasmic diffusion such that the passive component of
        d[C]/dt is rate_per_sec*([C]-[N])

        :param rate_per_sec: the max rate of passive diffusion in 1/(second*M)
        :return: None
        """
        self.max_passive_diffusion_rate_nmol_per_sec_per_M = \
            rate_per_sec * N_A * self.v_C_L  # convert per_M to per_nmol

    def set_params(self, **kwargs) -> None:
        """
        Sets attribute values for the transport simulation given parameter names and values

        :return: None
        """
        for param, value in kwargs.items():
            assert hasattr(self, param)
            setattr(self, param, value)

    def set_NPC_dock_sites(self, n_NPCs: int,
                           n_dock_sites_per_NPC: int  # TODO: this may depend on molecule size
                           ) -> None:
        """
        Sets the total number of dock sites for cargo-importin complex on NPCs

        :param n_NPCs: total number of nuclear pore complexes
        :param n_dock_sites_per_NPC: number of dock sites for cargo-importin complex per NPC
        :return: None
        """
        self.NPC_dock_sites = n_NPCs * n_dock_sites_per_NPC

    def _init_simulation_parameters(self, **kwargs) -> None:
        """
        Adds some default attributes to the simulation while also adding in some user-defined attributes from
        user-defined function parameters. Runs when running the init method

        :return: None
        """
        # TODO: add all simulation parameters here with proper units
        self.dt_sec = 1e-3  # simulation time step
        # NPC dock capacity: #TODO
        self.set_NPC_dock_sites(n_NPCs=200,
                                n_dock_sites_per_NPC=500)  # (maximal estimate from Timney et al. 2016 paper for yeast, nsites are a rule of thumb)
        # Rates:  # TODO: change nmol to nmolec - to prevent confusion between moles and molecules
        self.rate_complex_to_NPC_per_free_site_per_sec_per_M = 0.5e+6
        self.fraction_complex_NPC_traverse_per_sec = 1e+7  # fraction of complexes that go from one side of the NPC to the other per sec
        self.fraction_complex_NPC_to_free_N_per_M_GTP_per_sec = 0.005e+6  # TODO: this is doubled relative to complex_N to free_N
        self.fraction_complex_N_to_free_N_per_M_GTP_per_sec = 0.005e+6
        self.fraction_complex_NPC_to_complex_N_C_per_sec = 1.0  # Leakage parameter
        # TODO: look into some missing rates here (GTP_C to GTP_N, GDP_C to GTP_C (these may not happen, but need to investigate))
        self.rate_GDP_N_to_GTP_N_per_sec = 200.0
        self.rate_GTP_N_to_GDP_N_per_sec = 0.2
        self.rate_GTP_C_to_GDP_C_per_sec = 500.0
        self.rate_GTP_N_to_GTP_C_per_sec = 0.15
        self.rate_GDP_C_to_GDP_N_per_sec = 0.2
        self.rate_GDP_N_to_GDP_C_per_sec = 0.2
        self.rate_complex_to_free_per_sec = 0.05
        self.rate_free_to_complex_per_sec = 0.10  # assuming importins are not rate limiting in either cytoplasm aor nucleus and have identical concentration
        self.passive_competition_weight = 0.0  # a number between 0.0 and 1.0 quantifying the weight of competition # TODO: this could be a flag
        self.max_passive_diffusion_rate_nmol_per_sec_per_M = 20000  # as the name suggests, without accounting for competition effects # TODO: in future, a single number for both import and export that is independent of C/N volumes, # of NPCs etc
        self.bleach_volume_L_per_sec = 1.0e-15  # cytoplasmic cargo volume being bleached per second
        self.bleach_start_time_sec = np.inf  # no bleaching by default
        self.Ran_cell_M = 20e-6
        self.init_cargo_cytoplasm_M = 50e-6
        self.init_fraction_bound = 0.0
        self.set_params(**kwargs)

    def __init__(self, v_C_L: float = 55.85e-15, v_N_L: float = 4.35e-15, **kwargs):
        """
        Set initial state of the simulation
        :param v_C_L: the cytoplasmic volume in liters
        :param v_N_L: the nuclear volume in liters
        """
        self._init_simulation_parameters(**kwargs)
        self.sim_time_sec = 0.0
        self.nmol = {}  # number of molecules of various species
        # Cell geometry:
        self.v_C_L = v_C_L  # Cytoplasmic volume in L
        self.v_N_L = v_N_L  # Nuclear volume in L
        #        self.v_C_L= 10e-15 # Cytoplasmic volume in L
        #        self.v_N_L= 5e-15 # Nuclear volume in L
        self.reset_cargo_concentration(self.init_cargo_cytoplasm_M, self.init_fraction_bound)
        # import export per dt_sec
        self.nmol[
            "nuclear_importL_per_sec"] = 0  # molar rate of raw import to the nucleus, given cytoplasmic concentration (dN/dt=rate*[C])
        self.nmol[
            "nuclear_exportL_per_sec"] = 0  # molar rate of raw import to the nucleus, given cytoplasmic concentration (dN/dt=rate*[C])
        self.nmol[
            "nuclear_importU_per_sec"] = 0  # molar rate of raw import to the nucleus, given cytoplasmic concentration (dN/dt=rate*[N])
        self.nmol[
            "nuclear_exportU_per_sec"] = 0  # molar rate of raw import to the nucleus, given cytoplasmic concentration (dN/dt=rate*[N])
        # Ran in all:
        self.set_RAN_distribution(Ran_cell_M=self.Ran_cell_M,
                                  # total physiological concentration of Ran # TODO: check in the literature
                                  parts_GTP_N=1000,
                                  parts_GTP_C=1,
                                  parts_GDP_N=1,
                                  parts_GDP_C=1000)

    ##########################
    # Transitions calculators:
    ########################

    @register_update()
    def get_nmol_complex_NPC_to_free_N(self, T_list: list) -> None:
        """                                                                                                        
        Computes the number of complexed cargo molecules (labeled and unlabeled) released from the NPC into the nucleus
        to become free cargo molecules over the dt time step and updates the transition list accordingly
        (Note: It is assumed each undocking leads to export of a single RanGTP molecule out of the nucleus and list is
               updated accordingly)

        :param T_list: a list of tuples (src, dst, nmol), each representing a transfer of a number of molecules (nmol)
                       from a molecular species source (src) to the molecular species destination (dst)
        :return: None
        """
        #return float(int(np.power(nmol_GTP_N/max_RAN, 5)*nmol_NPC))
        f = self.fraction_complex_NPC_to_free_N_per_M_GTP_per_sec * self.get_concentration_M("GTP_N") * self.dt_sec
        n_GTP = 0
        for suffix in ["import", "export"]:
            for label in ["L", "U"]:
                src = f"complex{label}_NPC_N_{suffix}"
                dst = f"free{label}_N"
                n = f * self.nmol[src]
                n_GTP += n
                register_move_nmol(T_list,
                                   src=src,
                                   dst=dst,
                                   nmol=n)
        register_move_nmol(T_list,
                           src="GTP_N",
                           dst="GTP_C",
                           nmol=n_GTP)

    @register_update()
    def get_nmol_complex_N_to_free_N(self, T_list: list) -> None:
        """
        Computes the number of complexed cargo molecules (labeled and unlabeled) that disassemble in the nucleus over
        the dt time step and updates the transition list accordingly

        #TODO: figure out why GTP is updated here. I think this is passive diffusion and dissociation
        Note: it is assumed each GTP-dependent undocking leads to export of a single RanGTP molecule instantaneously

        :param T_list: a list of tuples (src, dst, nmol), each representing a transfer of a number of molecules (nmol)
                       from a molecular species source (src) to the molecular species destination (dst)
        :return: None
        """
        f_GTP = self.fraction_complex_N_to_free_N_per_M_GTP_per_sec * self.get_concentration_M("GTP_N") * self.dt_sec
        f_no_GTP = (1.0 - f_GTP) * self.rate_complex_to_free_per_sec * self.dt_sec  # non-GTP dependent undocking from remaining fraction
        f = f_GTP + f_no_GTP
        nL = f * self.nmol["complexL_N"]
        nU = f * self.nmol["complexU_N"]
        n_GTP = f_GTP * (self.nmol["complexL_N"] + self.nmol["complexU_N"])
        #     print("n {} GTP_N {} complex_N {}".format(n, self.nmol["GTP_N"], self.nmol["complex_N"]))
        assert n_GTP <= self.nmol["GTP_N"] and nL <= self.nmol["complexL_N"] and nU <= self.nmol["complexU_N"]
        register_move_nmol(T_list,
                           src="complexL_N",
                           dst="freeL_N",
                           nmol=nL)
        register_move_nmol(T_list,
                           src="complexU_N",
                           dst="freeU_N",
                           nmol=nU)
        register_move_nmol(T_list,
                           src="GTP_N",
                           dst="GTP_C",
                           nmol=n_GTP)

    @register_update()
    def get_nmol_GDP_N_to_GTP_N(self, T_list: list) -> None:
        """
        Computes the net number of GDP molecules in the nucleus converted to GTP over the dt time step and updates the
        transition list accordingly

        :param T_list: a list of tuples (src, dst, nmol), each representing a transfer of a number of molecules (nmol)
                       from a molecular species source (src) to the molecular species destination (dst)
        :return: None
        """
        n1 = self.rate_GDP_N_to_GTP_N_per_sec * self.nmol["GDP_N"] * self.dt_sec
        n2 = self.rate_GTP_N_to_GDP_N_per_sec * self.nmol["GTP_N"] * self.dt_sec
        n = n1 - n2
        register_move_nmol(T_list,
                           src="GDP_N",
                           dst="GTP_N",
                           nmol=n)

    @register_update()
    def get_nmol_GTP_C_to_GDP_C(self, T_list: list) -> None:
        """
        Computes the number of GTP molecules in the cytoplasm converted to GDP and updates the transition list
        accordingly

        :param T_list: a list of tuples (src, dst, nmol), each representing a transfer of a number of molecules (nmol)
                       from a molecular species source (src) to the molecular species destination (dst)
        :return: None
        """
        n = self.rate_GTP_C_to_GDP_C_per_sec * self.nmol["GTP_C"] * self.dt_sec
        register_move_nmol(T_list,
                           src="GTP_C",
                           dst="GDP_C",
                           nmol=n)

    @register_update()
    def get_nmol_GTP_N_to_GTP_C(self, T_list: list) -> None:
        """
        Computes the number of GTP molecules exported from the nucleus to the cytoplasm over the dt time step and
        updates the transition list accordingly

        :param T_list: a list of tuples (src, dst, nmol), each representing a transfer of a number of molecules (nmol)
                       from a molecular species source (src) to the molecular species destination (dst)
        :return: None
        """
        n = self.rate_GTP_N_to_GTP_C_per_sec * self.nmol["GTP_N"] * self.dt_sec
        register_move_nmol(T_list,
                           src="GTP_N",
                           dst="GTP_C",
                           nmol=n)

    @register_update()
    def get_nmol_GDP_C_to_GDP_N(self, T_list: list) -> None:
        """
        Computes the net number of GDP molecules imported from the cytoplasm into the nucleus over the dt time step and
        updates the transition list accordingly

        :param T_list: a list of tuples (src, dst, nmol), each representing a transfer of a number of molecules (nmol)
                       from a molecular species source (src) to the molecular species destination (dst)
        :return: None
        """
        n1 = self.rate_GDP_C_to_GDP_N_per_sec * self.nmol["GDP_C"] * self.dt_sec
        n2 = self.rate_GDP_N_to_GDP_C_per_sec * self.nmol["GDP_N"] * self.dt_sec
        n = n1 - n2
        register_move_nmol(T_list,
                           src="GDP_C",
                           dst="GDP_N",
                           nmol=n)

    @register_update()
    def get_nmol_complex_C_to_free_C(self, T_list: list) -> None:
        """
        Computes the number of cargo-importin complexes (both labeled and unlabeled) in the cytoplasm that unbind
        importin over the dt time step and updates the transition list accordingly

        :param T_list: a list of tuples (src, dst, nmol), each representing a transfer of a number of molecules (nmol)
                       from a molecular species source (src) to the molecular species destination (dst)
        :return: None
        """
        f = self.rate_complex_to_free_per_sec * self.dt_sec
        nL = f * self.nmol["complexL_C"]
        nU = f * self.nmol["complexU_C"]
        assert (nL <= self.nmol["complexL_C"])
        assert (nU <= self.nmol["complexU_C"])
        register_move_nmol(T_list,
                           src="complexL_C",
                           dst="freeL_C",
                           nmol=nL)
        register_move_nmol(T_list,
                           src="complexU_C",
                           dst="freeU_C",
                           nmol=nU)

    @register_update()
    def get_nmol_free_C_to_complex_C(self, T_list: list) -> None:  # assume importin is not rate limiting
        """
        Computes the number of the cargo molecules in the cytoplasm (labeled and unlabeled) that bind to importin over
        the dt time step and updates the transition list accordingly

        :param T_list: a list of tuples (src, dst, nmol), each representing a transfer of a number of molecules (nmol)
                       from a molecular species source (src) to the molecular species destination (dst)
        :return: None
        """
        f = self.rate_free_to_complex_per_sec * self.dt_sec
        nL = f * self.nmol["freeL_C"]
        nU = f * self.nmol["freeU_C"]
        assert (nL <= self.nmol["freeL_C"])
        assert (nU <= self.nmol["freeU_C"])
        register_move_nmol(T_list,
                           src="freeL_C",
                           dst="complexL_C",
                           nmol=nL)
        register_move_nmol(T_list,
                           src="freeU_C",
                           dst="complexU_C",
                           nmol=nU)

    @register_update()
    def get_nmol_free_N_to_complex_N(self, T_list: list) -> None:  # assume importin is not rate limiting
        """
        Computes the number of the cargo molecules in the nucleus (labeled and unlabeled) that bind to importin over the
        dt time step and updates the transition list accordingly

        :param T_list: a list of tuples (src, dst, nmol), each representing a transfer of a number of molecules (nmol)
                       from a molecular species source (src) to the molecular species destination (dst)
        :return: None
        """
        f = self.rate_free_to_complex_per_sec * self.dt_sec
        nL = f * self.nmol["freeL_N"]
        nU = f * self.nmol["freeU_N"]
        assert (nL <= self.nmol["freeL_N"])
        assert (nU <= self.nmol["freeU_N"])
        register_move_nmol(T_list,
                           src="freeL_N",
                           dst="complexL_N",
                           nmol=nL)
        register_move_nmol(T_list,
                           src="freeU_N",
                           dst="complexU_N",
                           nmol=nU)

    @register_update()
    def get_free_N_to_free_C(self, T_list: list) -> None:  # passive
        """
        Computes the number of unbound molecules that passively export to the cytoplasm and import to the nucleus over
        the dt time step (both labeled and unlabeled) and updates the transition list accordingly

        :param T_list: a list of tuples (src, dst, nmol), each representing a transfer of a number of molecules (nmol)
                       from a molecular species source (src) to the molecular species destination (dst)
        :return: None

        # COMMENT: a proper treatment of this would depend on ratio between nuclear 
        # and cytoplasmic volumes, number of NPCs etc. - here we ignore this for now
        # - we can change it in future based on theoretical equations of passive diffusion
        """
        # TODO: doublecheck the comment above and below. I think they addressed parts of these comments
        # Comment: competition is assumed to have zero effect at this time
        bound_dock_sites = sum([self.nmol[key] for key in self.nmol if "NPC" in key])
        fraction_bound_dock_sites_NPC = bound_dock_sites / self.NPC_dock_sites
        competition_multiplier = 1.0 - self.passive_competition_weight * fraction_bound_dock_sites_NPC
        f = self.max_passive_diffusion_rate_nmol_per_sec_per_M \
            * competition_multiplier \
            * self.dt_sec
        nL_export = f * self.get_concentration_M("freeL_N")
        nL_import = f * self.get_concentration_M("freeL_C")
        nU_export = f * self.get_concentration_M("freeU_N")
        nU_import = f * self.get_concentration_M("freeU_C")

        register_move_nmol(T_list,
                           src="freeL_N",
                           dst="freeL_C",
                           nmol=nL_export)
        register_move_nmol(T_list,
                           src="freeU_N",
                           dst="freeU_C",
                           nmol=nU_export)
        register_move_nmol(T_list,
                           src="freeL_C",
                           dst="freeL_N",
                           nmol=nL_import)
        register_move_nmol(T_list,
                           src="freeU_C",
                           dst="freeU_N",
                           nmol=nU_import)

    @register_update()
    def get_nmol_complex_N_C_to_complex_NPC(self, T_list: list) -> None:
        """
        Computes the number of cargo-importin complexed molecules (labeled and unlabeled) that bind to the NPC on the
        nuclear or cytoplasmic side from the nucleus and cytoplasm (respectively) over the dt time step and updates the
        transition list accordingly

        :param T_list: a list of tuples (src, dst, nmol), each representing a transfer of a number of molecules (nmol)
                       from a molecular species source (src) to the molecular species destination (dst)
        :return: None
        """
        bound_dock_sites = sum([self.nmol[key] for key in self.nmol if "NPC" in key])
        nmol_free_sites_NPC = (self.NPC_dock_sites - bound_dock_sites)
        f = nmol_free_sites_NPC * self.rate_complex_to_NPC_per_free_site_per_sec_per_M * self.dt_sec
        # TODO: debug - something is weird here (BR Dec 11,2020) (I think this is fixed but...)
        cL_N_M = self.get_concentration_M("complexL_N")
        cL_C_M = self.get_concentration_M("complexL_C")
        cU_N_M = self.get_concentration_M("complexU_N")
        cU_C_M = self.get_concentration_M("complexU_C")
        nL_N = f * cL_N_M
        nL_C = f * cL_C_M
        nU_N = f * cU_N_M
        nU_C = f * cU_C_M
        if CHECKS:
            assert_coeff = 2.0
            assert1_almost = (nL_N + nL_C + nU_N + nU_C <= assert_coeff * nmol_free_sites_NPC)
            assert2_almost = (nL_N + nU_N <= assert_coeff * (self.nmol["complexL_N"] + self.nmol["complexU_N"]))
            assert3_almost = (nL_C + nU_C <= assert_coeff * (self.nmol["complexL_C"] + self.nmol["complexU_C"]))
            if not (assert1_almost and assert2_almost and assert3_almost):
                assert1 = (nL_N + nL_C + nU_N + nU_C <= nmol_free_sites_NPC)
                assert2 = (nL_N + nU_N <= self.nmol["complexL_N"] + self.nmol["complexU_N"])
                assert3 = (nL_C + nU_C <= self.nmol["complexL_C"] + self.nmol["complexU_C"])
                print(self.nmol)
                print(f"f {f} dLabeled: N {nL_N} C {nL_C}, dUnlabeled: N {nU_N} C {nU_C}")
                assert assert1
                assert assert2
                assert assert3
                assert (assert1 and assert2 and assert3)
        register_move_nmol(T_list,
                           src="complexL_N",
                           dst="complexL_NPC_N_export",
                           nmol=nL_N)
        register_move_nmol(T_list,
                           src="complexL_C",
                           dst="complexL_NPC_C_import",
                           nmol=nL_C)
        register_move_nmol(T_list,
                           src="complexU_N",
                           dst="complexU_NPC_N_export",
                           nmol=nU_N)
        register_move_nmol(T_list,
                           src="complexU_C",
                           dst="complexU_NPC_C_import",
                           nmol=nU_C)

    @register_update()
    def get_nmol_complex_NPC_traverse(self, T_list: list) -> None:
        """
        Computes the number of complexed cargo molecules (labeled and unlabeled) passing from the nuclear side and
        cytoplasmic side of the NPC (or vice versa) and updates the transition list accordingly

        :param T_list: a list of tuples (src, dst, nmol), each representing a transfer of a number of molecules (nmol)
                       from a molecular species source (src) to the molecular species destination (dst)
        :return: None
        """
        # Note - Delta is the solution to the differential equation
        # for bidirectional movement from one side to another very
        # similarly to Fick's law. If we have n1 molecules on one side
        # and n2 molecules on the other, using Delta=(n1-n2) and
        # tau-1.0/self.fraction_complex_NPC_traverse_per_sec, then
        # dDelta/dt=Delta/tau, the solution being Delta(t) =
        # Delta(t0)*exp(-t/tau). Let Delta_fractional=exp(-t/tau),
        # then the net fraction of molecules traversing from one side
        # to the other over dt_sec seconds is thus half of (1.0 -
        # Delta_fractional)
        # TODO: figure out why this one is different than the other ones
        Delta_fractional = np.exp(-self.fraction_complex_NPC_traverse_per_sec * self.dt_sec)
        f = 0.5 * (1 - Delta_fractional)
        #        f= min(f, 0.5) # Note: if f is larger than 0.5, traversal time through NPC >> dt_sec, so we assume it just equilibrates (it can't be rate limiting if all other processes are slow relative to dt_sec) - see TODO above

        for label in ["L", "U"]:
            for src in ["N", "C"]:
                dst = "C" if (src == "N") else "N"
                for suffix in ["import", "export"]:
                    tag_src = f"complex{label}_NPC_{src}_{suffix}"
                    tag_dst = f"complex{label}_NPC_{dst}_{suffix}"
                    n = f * self.nmol[tag_src]
                    register_move_nmol(T_list,
                                       src=tag_src,
                                       dst=tag_dst,
                                       nmol=n)

    @register_update()
    def get_nmol_complex_NPC_to_complex_N_C(self, T_list: list) -> None:
        """
        Computes the number of complexed cargo-importin (labeled and unlabeled) released from the nuclear and
        cytoplasmic ends of the NPC to the nucleus and cytoplasm, respectively, over the dt time step and updates the
        transition list accordingly

        :param T_list: a list of tuples (src, dst, nmol), each representing a transfer of a number of molecules (nmol)
                       from a molecular species source (src) to the molecular species destination (dst)
        :return: None
        """
        f = self.fraction_complex_NPC_to_complex_N_C_per_sec * self.dt_sec  # fractions are fine (conceptually, a random variable)

        for label in ["L", "U"]:
            for compartment in ["N", "C"]:
                for suffix in ["import", "export"]:
                    src = f"complex{label}_NPC_{compartment}_{suffix}"
                    dst = f"complex{label}_{compartment}"
                    n = f * self.nmol[src]
                    register_move_nmol(T_list,
                                       src=src,
                                       dst=dst,
                                       nmol=n)

    @register_update()
    def get_nmol_cargo_bleached(self, T_list: list) -> None:
        """
        Computes the number of fluorescent cargo molecules that have become bleached over the dt time step (both free
        and complexed) and updates the transition list accordingly

        :param T_list: a list of tuples (src, dst, nmol), each representing a transfer of a number of molecules (nmol)
                       from a molecular species source (src) to the molecular species destination (dst)
        :return: None
        """
        global N_A
        if self.sim_time_sec <= self.bleach_start_time_sec:
            return
        f = self.bleach_volume_L_per_sec \
            * N_A \
            * self.dt_sec
        c_freeL_C_M = self.get_concentration_M('freeL_C')
        n_free_C = f * c_freeL_C_M
        c_complexL_C_M = self.get_concentration_M('complexL_C')
        n_complex_C = f * c_complexL_C_M
        #print(f"Bleaching {n_free_C} free cargo molecules")
        #print(self.nmol["freeL_C"], f)
        assert (n_free_C <= self.nmol["freeL_C"])
        assert (n_complex_C <= self.nmol["complexL_C"])
        register_move_nmol(T_list,
                           src="freeL_C",
                           dst="freeU_C",
                           nmol=n_free_C)
        register_move_nmol(T_list,
                           src="complexL_C",
                           dst="complexU_C",
                           nmol=n_complex_C)

    ##########################
    # Individual update rules:
    ########################

    def get_nmol_T_summary(self, T_list: list) -> dict:
        """
        Summarize all transitions by consolidating the transition list into a species-change dictionary

        :param T_list: a list of tuples (src, dst, nmol), each representing a transition of a number of molecules (nmol)
                       from a molecular species source (src) to a molecular species destination (dst)
        :return: a dictionary mapping from molecular species to total change in number of molecules
        """
        T = {}
        for src, dst, nmol in T_list:
            if src in T:
                T[src] -= nmol
            else:
                T[src] = -nmol

            if dst in T:
                T[dst] += nmol
            else:
                T[dst] = nmol

        return T

    def get_import_export_summary(self, T_list: list) -> None:
        """
        Compute a summary of the gross number of cargo molecules imported and exported and update the values for import
        and export rates of the cargo molecule in the nmol dictionary.
        Should be called BEFORE transitions are updated

        :param T_list: a list of tuples (src, dst, nmol), each representing a transition of a number of molecules (nmol)
                       from a molecular species source (src) to a molecular species destination (dst)
        :return: None
        """
        # TODO this is a hack - make it better
        active_import = {"L": 0, "U": 0}
        passive_import = {"L": 0, "U": 0}
        active_export = {"L": 0, "U": 0}
        passive_export = {"L": 0, "U": 0}

        for src, dst, nmol in T_list:
            # we aren't interested in Ran
            if "GTP" in src or "GDP" in src:
                continue
            is_bleach = "L" in src and "U" in dst
            if is_bleach:
                continue

            is_labeled = "L" in src
            assert ((is_labeled and "L" in dst) or ((not is_labeled) and "U" in dst))
            label = "L" if is_labeled else "U"
            # active import/export:
            is_facilitated_transport = ("NPC" in src and "NPC" not in dst)
            is_passive_diffusion = ("free" in src and "free" in dst)
            if is_facilitated_transport:
                if "_C" in dst and "export" in src:
                    active_export[label] += nmol
                if "_N" in dst and "import" in src:
                    active_import[label] += nmol

            elif is_passive_diffusion:
                # passive export
                if "N" in src and "C" in dst:
                    passive_export[label] += nmol
                # passive import
                elif "C" in src and "N" in dst:
                    passive_import[label] += nmol

        for label in ["L", "U"]:
            total_N = sum([self.nmol[key] for key in self.nmol if "N" in key
                           and label in key
                           and "NPC" not in key
                           and "G" not in key])
            total_C = sum([self.nmol[key] for key in self.nmol if "C" in key
                           and label in key
                           and "NPC" not in key
                           and "G" not in key])
            if total_N == 0:
                self.nmol[f"nuclear_export{label}_per_sec"] = 0
            else:
                self.nmol[f"nuclear_export{label}_per_sec"] = \
                    (active_export[label] + passive_export[label]) / (
                            total_N * self.dt_sec)  # the rate at which a single nuclear molecule is imported, or equivalently, the ratio between d[N]/dt and [N]
            if total_C == 0:
                self.nmol[f"nuclear_import{label}_per_sec"] = 0
            else:
                cytoplasmic_import_rate_per_sec = \
                    (active_import[label] + passive_import[label]) / (
                            total_C * self.dt_sec)  # rate at which a single cytoplasmic molecule is imported, or equivalently, the ratio between d[C]/dt and [C]
                self.nmol[f"nuclear_import{label}_per_sec"] = \
                    cytoplasmic_import_rate_per_sec * (
                            self.get_v_C_L() / self.get_v_N_L())  # from d[C]/dt to d[N]/dt as a function of [C]

    def do_one_time_step(self) -> None:
        """
        Updates all state variables over a single time step

        :return: None
        """
        # Compute transitions:
        T_list = []
        for update_rule in self._update_funcs:
            update_rule(self, T_list)
        T = self.get_nmol_T_summary(T_list)
        #        print("Do_step")
        #        print(T_list)
        self.get_import_export_summary(T_list)

        # Update transitions:
        for key, value in T.items():
            if key not in self.nmol:
                raise ValueError(f"can't update non-existent molecular species {key}")
            self.nmol[key] += value
            if self.nmol[key] < 0:
                if self.nmol[key] > -0.001:
                    T[key] = 0.0
                else:
                    print(f"Negative key {key} value {self.nmol[key]} change {value}")
                    print(T)
                    print(self.nmol)
                    assert (self.nmol[key] >= 0)
                    # Update simulation clock:
        self.sim_time_sec += self.dt_sec

    def simulate(self, sim_time_sec: float, nskip_statistics: int = 1) -> dict:
        """
        Simulate for approximately (and at least) sim_time_sec seconds

        :param sim_time_sec: time interval (in seconds) for which the simulation is run
        :param nskip_statistics: time step interval to record data (e.g. 2 is every two time steps)
        :return: dictionary with that contain the values for each species at each time point frame
        """
        # Computes number of steps and frames
        nsteps = int(np.ceil(sim_time_sec / self.dt_sec))
        nframes = ((nsteps - 1) // nskip_statistics) + 1
        # Prepare statistics dictionary for all molecule types
        stats = {'time_sec': np.zeros(nframes)}
        for key in self.nmol.keys():
            stats[key] = np.zeros(nframes)
        for i in range(nsteps):
            self.do_one_time_step()
            if i % nskip_statistics == 0:
                si = i // nskip_statistics
                stats['time_sec'][si] = self.sim_time_sec
                for key, value in self.nmol.items():
                    stats[key][si] = value
        return stats

    ##########################
    # Debug utility functions
    ########################
    def get_total_RAN(self) -> float:
        """
        Returns the total number of molecules of RAN_GDP + RAN_GTP in the cell
        """
        RAN = self.nmol["GDP_C"] + self.nmol["GTP_C"] + self.nmol["GDP_N"] + self.nmol["GTP_N"]
        return RAN

    def get_total_cargoL_nmol(self) -> float:
        """
        Returns the total number of molecules of labeled cargo (bound and unbound) in the cell
        """
        def is_cargoL(s: str) -> bool:
            """
            Returns whether species contains labeled cargo (can be in complex or free)

            :param s: species tag
            """
            return s.startswith("freeL_") or s.startswith("complexL_")

        return sum([self.nmol[key] for key in self.nmol if is_cargoL(key)])

    def get_total_cargoU_nmol(self) -> float:
        """
        Returns the total number of molecules of unlabeled cargo (bound and unbound) in the cell
        """
        def is_cargoU(s: str) -> bool:
            """
            Returns whether species contains unlabeled cargo (can be in complex or free)

            :param s: species tag
            """
            return s.startswith("freeU_") or s.startswith("complexU_")

        return sum([self.nmol[key] for key in self.nmol if "U" in key])

    def get_total_cargo_nmol(self) -> float:
        """
        Returns the total number of molecules of cargo (labeled and unlabeled, bound and unbound) in the cell
        """
        return self.get_total_cargoL_nmol() + self.get_total_cargoU_nmol()
