import threading
import os
import warnings
import re
import myfilemanager as mfm
from collections import defaultdict
import numpy as np
import io
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import PchipInterpolator
from typing import Callable, Any, Union
from time import time
import matplotlib as mpl
import yaml
import pickle
from tqdm import tqdm

class PyECLOUDsim:
    def __init__(self, input_folder: str, sim_output_filename: str = "output.mat", params_yaml = None, load_sim_data: bool = True):
        if not os.path.exists(input_folder):
            raise IsADirectoryError("Provided input path path does not exist")
        if not params_yaml:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.params_yaml = os.path.join(script_dir, 'params.yaml')
        else:
            if os.path.exists(params_yaml):
                self.params_yaml = params_yaml
            else:
                raise FileNotFoundError("Params YAML file not found")
            
        self.input_folder = input_folder
        self.sim_output_filename = sim_output_filename
        self.sim_data_loaded = load_sim_data
        self.sim_data = None # Simulation output in object form
        # self.SEY = None # Value of SEY
        # self.intensity = None # Value of Intensity
        # self.beam_filling_pattern = None # Beam Filling Pattern used
        # self.photoemission_enabled = None # Determines whether photoemission is used or not
        # self.beam_energy = None # Beam energy in eV
        # self.magnet_conf = None # Magnetic field configuration
        sim_filenames = self.parse_values_from_files(["machine_param_file",
                                                      "secondary_emission_parameters_file",
                                                      "beam_parameters_file",
                                                      "progress_path"],
                                                      ["simulation_parameters.input"]                                                     
                                                      )
        self.machine_param_file = sim_filenames['machine_param_file']
        self.SEY_params_file = sim_filenames['secondary_emission_parameters_file']
        self.beam_params_file = sim_filenames['beam_parameters_file']
        with open(os.path.join(self.input_folder,sim_filenames['progress_path'])) as f:
            sim_progress = float(f.read().strip())
        if sim_progress < 0.9:
            warnings.warn(f"!----Simulation progress below 0.9. Simulation at {self.input_folder} might be incomplete.----!")
        self._load_sim_from_yaml_()
        # self._load_sim_()

    def dump(self, filename: str = "sim_db.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self,f)

    # This ensures that all attributes from sim_data are accessible from the main class
    def __getattr__(self, name):
        # Only called if the attribute wasn't found on self
        return getattr(self.sim_data, name)
    
    def _convert_data_to_dtype_(self, value, dtype_str, dtype_map: dict = {
                                    'float': float,
                                    'int': int,
                                    'str': str,
                                    'bool': lambda x: str(x).lower() in ['true', '1', 'yes'],
                                }):
        try:
            converter = dtype_map.get(dtype_str)
            if not converter:
                raise ValueError(f"Unsupported dtype: {dtype_str}")
            if value is not None:
                return converter(value)
            else:
                return None
        except Exception as e:
            raise ValueError(f"Error converting value '{value}' to type '{dtype_str}': {e}")


    def _load_sim_from_yaml_(self):
        with open(self.params_yaml, 'r') as file:
            params = yaml.safe_load(file)
        if self.sim_data_loaded:
            self.sim_data = mfm.myloadmat_to_obj(os.path.join(self.input_folder,self.sim_output_filename))
        param_names = params.keys()
        vars_to_load = [entry.get('var_name') for entry in params.values()]
        sim_params = self.parse_values_from_files(vars_to_load, is_internal = True)
        
        for param in param_names:
            attr_name = param.replace(" ", "_")
            setattr(self, attr_name, self._convert_data_to_dtype_(sim_params[params[param]["var_name"]], params[param]["dtype"]))
        
        return params, vars_to_load

    def _load_sim_(self):
        if self.sim_data_loaded:
            self.sim_data = mfm.myloadmat_to_obj(os.path.join(self.input_folder,self.sim_output_filename))
        sim_params = self.parse_values_from_files(["del_max",
                                                   "filling_pattern_file",
                                                   "fact_beam",
                                                   "photoem_flag",
                                                   "energy_eV",
                                                   "B_multip"])
        self.SEY = float(sim_params['del_max'])
        self.beam_filling_pattern = sim_params['filling_pattern_file']
        self.photoemission_enabled = bool(int(sim_params['photoem_flag']))
        self.beam_energy = float(sim_params['energy_eV'])
        self.intensity = float(sim_params['fact_beam'])
        self.magnet_conf = sim_params['B_multip']

    def parse_values_from_files(self,attribute_names: list,filenames: list = None, is_internal: bool = False):
        '''
        Get attributes from simulation files. attribute_names is a list of strings, filenames (optional) can be used to specify specific filenames to search
        '''
        
        if filenames is None:
            filenames = [self.machine_param_file,
                         self.SEY_params_file,
                         self.beam_params_file
                         ]
            
        def find_attribute(attribute_name, lines, attribute_vals, lock):
            pattern = re.compile(rf"^{attribute_name}\s*=\s*(.+)$")
            for line in lines:
                match = pattern.match(line.strip())
                if match:
                    value = match.group(1).strip()

                    # Remove surrounding quotes if present
                    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
                        value = value[1:-1]

                    with lock:
                        attribute_vals[attribute_name] = value
                    return
            if not is_internal:
                raise Exception(f"{attribute_name} not found in simulation files")
            else:
                with lock:
                    attribute_vals[attribute_name] = None

        lines = []
        for filename in filenames:
            with open(os.path.join(self.input_folder,filename)) as f:
                lines.extend(f.readlines())
        
        attribute_vals = {}
        lock = threading.Lock()
        threads = []

        for attribute_name in attribute_names:
            thread = threading.Thread(target=find_attribute, args=(attribute_name, lines, attribute_vals, lock))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()

        return attribute_vals
    
    def extract_beam_params(self):
        '''
        Returns an array containing beam parameters in the format:
        [number of trains,
        number of pattern inside train,
        number of bunches,
        intensity multiplier for bunches,
        number of empty slots,
        intensity multiplier for empty slots,
        number of empty slots between trains,
        intensity multiplier for empty slots]
        '''
        # Regex pattern to match the structure
        pattern = r"""
            ^\s*                            # Optional leading whitespace
            (\d+)\*                         # train_num (e.g., 2) followed by *
            \(\s*                           # Opening outer parenthesis
            (\d+)\*                         # repetitions (e.g., 4) followed by *
            \(\s*                           # Opening inner parenthesis
            (\d+)\*\[([\d.]+)\]             # filled_bunches * [intensity]
            \+                              # plus sign
            (\d+)\*\[([\d.]+)\]             # empty_bunches * [0.]
            \)\s*                           # Close inner parenthesis
            (?:\+\s*(\d+)\*\[([\d.]+)\])?   # Optional + outer empty slots * [value]
            \)\s*$                          # Close outer parenthesis and end
        """

        match = re.match(pattern, self.Beam_Filling_Pattern, re.VERBOSE)
        if not match:
            return None  # or raise ValueError("Pattern mismatch")

        groups = match.groups()

        # Convert extracted values
        result = [int(groups[0]), int(groups[1]), int(groups[2]), float(groups[3]),
                int(groups[4]), float(groups[5])]

        # Optional values
        if groups[6] is not None and groups[7] is not None:
            result.extend([int(groups[6]), float(groups[7])])
        else:
            result.extend([0, 0.0])

        return result

    def get_i_train_idx(self,train_num = -1):
        '''
        Get index of the i-th train for data saved per bunch passage. If unspecified, the index for the last train is returned.
        '''
        params = self.extract_beam_params()
        n_trains = params[0]
        n_reps_inside_train = params[1]
        n_filled = params[2]
        n_empty = params[4]
        n_sep_trains = params[6]
        train_length = n_reps_inside_train*(n_filled+n_empty) + n_sep_trains
        if train_num == -1:
            train_num = n_trains
        else:
            if train_num > n_trains:
                raise ValueError("Train index must not exceed number of total trains. Trains are considered 1-indexed")
        return ((train_num-1)*train_length) - 1
    
    def get_i_train_n_last_passages(self, n_indices: int = 6, train_num = -1):
        '''
        Get the indices for the last n bunch passages for the i-th train. If unspecified, the indices for the last train is returned.
        '''
        params = self.extract_beam_params()
        n_trains = params[0]
        n_reps_inside_train = params[1]
        n_filled = params[2]
        n_empty = params[4]
        n_sep_trains = params[6]
        train_length = n_reps_inside_train*(n_filled+n_empty) + n_sep_trains
        if train_num == -1:
            train_num = n_trains
        else:
            if train_num > n_trains:
                raise ValueError("Train index must not exceed number of total trains. Trains are considered 1-indexed")
        last_idx = ((train_num)*train_length) - n_sep_trains - n_empty
        return (last_idx - n_indices, last_idx)
    
    def get_i_train_timestep_idx(self, train_num = -1):
        '''
        Get index of the i-th train for data saved every timestep. If unspecified, the index for the last train is returned.
        '''
        params = self.extract_beam_params()
        reference_bunch_list = self.t_hist
        reference_timestep_list = self.t
        n_trains = params[0]
        n_reps_inside_train = params[1]
        n_filled = params[2]
        n_empty = params[4]
        n_sep_trains = params[6]
        train_length = n_reps_inside_train*(n_filled+n_empty) + n_sep_trains
        num_chunks = reference_bunch_list.shape[0]
        chunk_size = reference_timestep_list.shape[0] / num_chunks
        if train_num == -1:
            train_num = n_trains
        else:
            if train_num > n_trains:
                raise ValueError("Train index must not exceed number of total trains. Trains are considered 1-indexed")
        return int((((train_num-1)*train_length))*chunk_size - 1)
    
    def get_i_train_last_bunch_start(self, train_num = -1, last_timestep_idx = 5):
        '''
        Get index of the i-th train for data saved every timestep. If unspecified, the index for the last train is returned.
        '''
        params = self.extract_beam_params()
        reference_bunch_list = self.t_hist
        reference_timestep_list = self.t
        n_trains = params[0]
        n_reps_inside_train = params[1]
        n_filled = params[2]
        n_empty = params[4]
        n_sep_trains = params[6]
        train_length = n_reps_inside_train*(n_filled+n_empty) + n_sep_trains
        num_chunks = reference_bunch_list.shape[0]
        chunk_size = reference_timestep_list.shape[0] / num_chunks
        if train_num == -1:
            train_num = n_trains
        else:
            if train_num > n_trains:
                raise ValueError("Train index must not exceed number of total trains. Trains are considered 1-indexed")
        last_idx = int((((train_num-1)*train_length) + (n_reps_inside_train-1)*(n_filled+n_empty) + n_filled)*chunk_size)
        return (last_idx, last_idx + last_timestep_idx)
    
    def timestep_list_to_bunch_list(self, list_to_be_rescaled, mode="sum"):
        '''
        Convert list saved for every timestep to list saved for every bunch passage.

        Parameters:
            - list_to_be_rescaled: np.array of values per timestep
            - mode: "sum" or "avg" — controls aggregation function per bunch
        '''

        if mode not in ("sum", "avg"):
            raise ValueError("mode must be either 'sum' or 'avg'")

        reference_bunch_list = self.t_hist
        reference_timestep_list = self.t
        num_chunks = reference_bunch_list.shape[0]
        chunk_size = reference_timestep_list.shape[0] / num_chunks

        rescaled_list = np.array([
            list_to_be_rescaled[int(i*chunk_size):int((i+1)*chunk_size)].sum()
            if mode == "sum" else
            list_to_be_rescaled[int(i*chunk_size):int((i+1)*chunk_size)].mean()
            for i in range(num_chunks)
        ])
        
        return rescaled_list
    
    def calculate_heat_load_per_bunch(self, T_rev: float = 88.9e-6, unit: str = "mW"):
        '''
        Calculate heat load from simulation. 

        Parameters:
            - T_rev: Revolution frequency in seconds
            - unit: "mW", "W" or "eV" — unit for heatload. The option "eV" corresponds to eV/s units
        '''
        if unit not in ("mW", "W", "eV"):
            raise ValueError("Available units for function are 'W', 'mW' and 'eV'")
        qe = 1.60217657e-19

        unit_multiplier = {
            "eV" : 1,
            "mW" : 1000*qe,
            "W"  : qe
        }

        params = self.extract_beam_params()
        n_reps_inside_train = params[1]
        n_filled = params[2]
        bunch_num_in_train = n_reps_inside_train*n_filled

        return float(unit_multiplier[unit]*np.sum(self.sim_data.En_imp_eV_time[self.get_i_train_timestep_idx():])/(bunch_num_in_train*T_rev))
    
    def get_horizontal_nel_hist_i_passage(self, train_num = -1, n_indices: int = 6, bin_height: float = 1.800000e-02):
        bunch_passage_start, bunch_passage_end = self.get_i_train_n_last_passages(train_num=train_num, n_indices=n_indices)
        horizontal_hist = np.mean(self.nel_hist[bunch_passage_start:bunch_passage_end],axis = 0)
        bin_width = self.xg_hist[1] - self.xg_hist[0]
        horizontal_hist = horizontal_hist/(bin_width*bin_height)
        return (self.xg_hist, horizontal_hist)



class PyECLOUDParameterScan:
    def __init__(self, simulations_path: str, params_yaml = None , sim_output_filename: str = "output.mat", force_rebuild = False):
        if not os.path.exists(simulations_path):
            raise IsADirectoryError("Provided simulations path does not exist")
        if (not force_rebuild) and os.path.exists(os.path.join(simulations_path,"PyECLOUDparamscan.db")):
            with open(os.path.join(simulations_path,"PyECLOUDparamscan.db"), "rb") as f:
                state = pickle.load(f)
                self.__dict__.update(state)
                self.simulations_path = os.path.abspath(simulations_path)
        else:
            self.simulations_path = os.path.abspath(simulations_path)
            self.sim_output_filename = sim_output_filename
            self.sims_per_parameter = {}
            if not params_yaml:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                self.params_yaml = os.path.join(script_dir, 'params.yaml')
            else:
                if os.path.exists(params_yaml):
                    self.params_yaml = params_yaml
                else:
                    raise FileNotFoundError("Params YAML file not found")
            
            self.params_dict = self.read_yaml_to_dict(self.params_yaml)
            self.available_params = defaultdict(list)
            n_sims = self._find_sim_folders_and_extract_params_yaml_()
            # Convert intensity to smaller range while keeping absolute value
            self.scaled_vals = []

            for param in self.params_dict.keys():
                if self.params_dict[param]["dtype"] in ["float", "int"]:
                    if self.params_dict[param]["convert_to_pow"]:
                        attr_name = param.replace(" ", "_")+"_Pow"
                        numeric_keys = [k for k in self.sims_per_parameter[param].keys() if k is not None]
                        setattr(self, attr_name, int(np.log10(np.array(numeric_keys).mean())))
                        keys = list(self.sims_per_parameter[param].keys())
                        for key in keys:
                            if key is not None:
                                self.sims_per_parameter[param][float(key/(np.pow(10.0,getattr(self, attr_name))))] = self.sims_per_parameter[param].pop(key)
                        self.scaled_vals.append(param)
                        

            print(f"Found {n_sims} simulations in {simulations_path}. Available parameters for analysis:")

            # Key used for sorting numerical values
            def try_numeric(val):
                try:
                    return float(val)
                except ValueError:
                    return val
                except TypeError:
                    return -float("inf")
                
            for key in self.sims_per_parameter.keys():
                available_vals = list(self.sims_per_parameter[key].keys())
                if self.params_dict[key]["dtype"] in ["float","int"]:
                    available_vals.sort(key=try_numeric)
                if len(available_vals) > 1:
                    attr_name = key.replace(" ", "_")+"_Pow"
                    print(f"-'{key}'. With available values:{' (scaled by 1e%d)' % getattr(self,attr_name) if key in self.scaled_vals else ''}(Data Type: {self.params_dict[key]['dtype']})")
                    
                    for val in available_vals:
                        print(f"    ->{val}")
                        self.available_params[key].append(val)
            
            buffer = io.StringIO()
            sys_stdout = sys.stdout
            sys.stdout = buffer
            try:
                self.print_available_params()
            finally:
                sys.stdout = sys_stdout  # Reset stdout
            
            self.print_available_params_str = buffer.getvalue()

            with open(os.path.join(simulations_path,"PyECLOUDparamscan.db"), "wb") as f:
                self.simulations_path = None
                pickle.dump(self.__dict__, f)

            self.simulations_path = os.path.abspath(simulations_path)

    
    def _find_sim_folders_and_extract_params_yaml_(self):
        n_sims = 0
        param_names = self.params_dict.keys()
        for param in param_names:
            self.sims_per_parameter[param] = defaultdict(set)
        for root, dirs, files in tqdm(os.walk(self.simulations_path)):
            if "simulation_parameters.input" in files and self.sim_output_filename in files:
                sim = PyECLOUDsim(root, params_yaml=self.params_yaml, load_sim_data = False)
                for param in param_names:
                    attr_name = param.replace(" ", "_")
                    self.sims_per_parameter[param][getattr(sim,attr_name)].add(os.path.relpath(root,self.simulations_path))
                n_sims += 1
        return n_sims
    
    # def _find_sim_folders_and_extract_params_(self):
    #     n_sims = 0
    #     for root, dirs, files in os.walk(self.simulations_path):
    #         if "simulation_parameters.input" in files and self.sim_output_filename in files:
    #             sim = PyECLOUDsim(root, params_yaml=self.params_yaml, load_sim_data = False)
    #             self.sims_per_parameter["SEY"][sim.SEY].add(root)
    #             self.sims_per_parameter["Intensity"][sim.intensity].add(root)
    #             self.sims_per_parameter["Magnet Configuration"][sim.magnet_conf].add(root)
    #             self.sims_per_parameter["Beam Filling Pattern"][sim.beam_filling_pattern].add(root)
    #             self.sims_per_parameter["Photoemission Enabled"][sim.photoemission_enabled].add(root)
    #             self.sims_per_parameter["Beam Energy"][sim.beam_energy].add(root)
    #             n_sims += 1
    #     return n_sims
    
    def get_simulation(self, sim_params: Union[str, dict], is_internal: bool = False):
        '''
        Find simulations with desired parameters. 

        Parameters:
            - sim_params:   Dictionary or link to .yaml file containing parameters of the desired simulation 
            - is_internal:  Not to be used. Determines whether an error is returned for non-existent simulations and supresses output
        '''

        if isinstance(sim_params, str):
            if sim_params.endswith(".yaml"):
                sim_params = self.read_yaml_to_dict(sim_params)
            else:
                raise ValueError("sim_params can either be a .yaml file or a dict")
        elif isinstance(sim_params, dict):
            pass
        else:
            raise ValueError("sim_params can either be a .yaml file or a dict")
        
        scan_params = list(sim_params.keys())
        param_num = len(scan_params)
        try:
            sim_loc = None
            if param_num == 1:
                sim_loc = self.sims_per_parameter[scan_params[0]][sim_params[scan_params[0]]]
            elif param_num == 2:
                sim_loc = self.sims_per_parameter[scan_params[0]][sim_params[scan_params[0]]].intersection(self.sims_per_parameter[scan_params[1]][sim_params[scan_params[1]]])
            elif param_num > 2:
                sim_loc = self.sims_per_parameter[scan_params[0]][sim_params[scan_params[0]]].intersection(self.sims_per_parameter[scan_params[1]][sim_params[scan_params[1]]])
                for i in range(2,param_num):
                    sim_loc = sim_loc.intersection(self.sims_per_parameter[scan_params[i]][sim_params[scan_params[i]]])
            
            if sim_loc == set():
                if is_internal:
                    return None
                else:
                    raise Exception(f"No simulation found for the provided parameters ({sim_params}). Please check provided parameters and their data types.")
            else:
                sim_loc_final = str(sim_loc.pop())
                if sim_loc == set():
                    if not is_internal:
                        print(f"Simulation found at {sim_loc_final}")
                    return PyECLOUDsim(os.path.join(self.simulations_path, sim_loc_final), sim_output_filename = self.sim_output_filename)
                else:
                    sim_loc.add(sim_loc_final)
                    raise Exception(f"Specified parameters do not uniquely define simulation. \n Available simulations for specified parameters in {self.simulations_path} are: {sim_loc}\n All available parameters are: \n{self.print_available_params_str}")
        except KeyError:
            raise Exception(f"Check that provided parameters are in available parameters for plotting. All available parameters are: \n{self.print_available_params_str}")
    
    def get_param_values(self, param: str):
        if param not in self.available_params.keys():
            raise ValueError("Param must be in available params for plotting")
        return self.available_params[param]

    def print_available_params(self):
        for key in self.available_params.keys():
            attr_name = key.replace(" ", "_")+"_Pow"
            print(f"-'{key}'. With available values:{' (scaled by 1e%d)' % getattr(self,attr_name) if key in self.scaled_vals else ''} (Data Type: {self.params_dict[key]['dtype']})")
            for val in self.available_params[key]:
                print(f"    ->{val}")
    
    def get_value_units_dict_tex(self):
        val_units = {}
        for val in self.params_dict.keys():
            if self.params_dict[val]["unit"]:
                unit = rf"{{{self.params_dict[val]['unit']}}}"
                unit_tex = r"\mathrm" + unit.replace(" ", r"\,")
            else:
                unit_tex = None
            if val in self.scaled_vals:
                pow_attr_name = val.replace(" ", "_")+"_Pow"
                val_units[val] = rf"\times 10^{{{getattr(self,pow_attr_name)}}}\,\mathrm{{{unit_tex}}}"
            else:
                val_units[val] = unit_tex
        return val_units

    def get_value_units_dict(self):
        val_units = {}
        for val in self.params_dict.keys():
            if val in self.scaled_vals:
                pow_attr_name = val.replace(" ", "_")+"_Pow"
                val_units[val] = f"* 10^{getattr(self,pow_attr_name)} {self.params_dict[val]['unit']}"
            else:
                val_units[val] = self.params_dict[val]["unit"]
        return val_units

    def read_yaml_to_dict(self, yaml_filepath):
        with open(yaml_filepath, 'r') as file:
            yaml_dict = yaml.safe_load(file)
        return yaml_dict

    # def _get_curves_with_colors_(self,curves_to_plot: dict, curves_is_str: bool, curve_vals: list = None, 
    #                             curve_colors: dict = None, cmap_min_offset: float = 0, cmap_max_offset: float = 0):
    #     if curve_colors is None:
    #         if curves_is_str and (isinstance(curve_vals[0],float) or isinstance(curve_vals[0],int)):
    #             norm = mcolors.Normalize(vmin=min(curve_vals) + cmap_min_offset, vmax=max(curve_vals) + cmap_max_offset)
    #             for curve_name, data in curves_to_plot.items():
    #                 data["color"] = cmap(norm(curve_name))
    #         else:
    #             all_avgs = [d["avg"] for d in curves_to_plot.values()]
    #             norm = mcolors.Normalize(vmin=min(all_avgs) + cmap_min_offset, vmax=max(all_avgs) + cmap_max_offset)
    #             for curve_name, data in curves_to_plot.items():
    #                 data["color"] = cmap(norm(data["avg"]))
    #             curves_to_plot = dict(sorted(curves_to_plot.items(), key=lambda item: item[1]['avg'], reverse=True))
    #     else:
    #         for curve_name, data in curves_to_plot.items():
    #             data["color"] = curve_colors[curve_name]
    #     return curves_to_plot

    def plot_simulation_result_vs_attrib(self, result_func: Callable[[Any], float], x_axis: str, curves: Union[str, dict], 
                       common_params: dict = {}, attrib_name: str = None, attrib_unit: str = None, x_axis_vals: list = None, curve_vals: list = None,
                       usetex: bool = True, global_fontsize: float = 18, curve_colors: dict = None,
                       use_interp: bool = True, interp_linspace_size: int = 300, show_datapoints: bool = True, lw: float = 2,
                       plot_figsize : tuple = (10,5), cmap = plt.cm.magma, cmap_min_offset: float = 0, cmap_max_offset: float = 0 ,
                       val_units: dict = None, show_legend: bool = True, legend_title: str = None, legend_bbox_to_anchor: tuple = (1.04, 0.5), legend_loc: str = "center left",
                       left_lim: float = None, right_lim: float = None, bottom_lim: float = 0, top_lim: float = None,
                       title: str = None, title_pad: float = 20, title_fontsize: float = 20,
                       xlabel: str = None, xlabel_pad: float = 10, xlabel_fontsize: float = 20,
                       ylabel: str = None, ylabel_pad: float = 10, ylabel_fontsize: float = 20,
                       grid: str = "minor", grid_major_linestyle: str = "-", grid_major_linewidth: float = 0.75,
                       grid_minor_linestyle: str = ":", grid_minor_linewidth: float = 0.5,
                       savefig: bool = False, output_filename: str = None, dpi: int = 300, show: bool = True, save_folder: str = "./",
                       returnfig: bool = False, round_xvals: int = 5, round_curvevals: int = 5
                       ):
        mpl.rcParams.update(mpl.rcParamsDefault)
        if not (isinstance(self.available_params[x_axis][0], float) or isinstance(self.available_params[x_axis][0], int)):
            raise ValueError("Values on x_axis cannot be non numeric")
        
        curves_dict = {}
        if isinstance(curves,str):
            if curves.endswith(".yaml"):
                curves_dict = self.read_yaml_to_dict(curves)
                curves_is_str = False
            else:
                if curve_vals is None:
                    curve_vals = self.available_params[curves]
                    curves_dict = {v: {curves : v} for v in curve_vals}
                else:
                    curves_dict = {round(v, round_curvevals): {curves : round(v, round_curvevals)} for v in curve_vals}
                curves_is_str = True
        elif isinstance(curves,dict):
            if curve_vals:
                raise ValueError("If curve_vals is specified, 'curves' must uniquely define a parameter that can be iterable")
            curves_dict = curves
            curves_is_str = False
        else:
            raise ValueError("'curves' must either define a specific parameter as 'str' or a group of parameters as 'dict'")

        if usetex:
            plt.rcParams.update({
                "text.usetex": True,        # Use LaTeX to render all text
                "font.family": "serif",     # Use serif fonts (Computer Modern is default)
                "font.serif": ["Computer Modern"],  # Optionally specify which serif font
                "font.size" : global_fontsize
            })

            if not val_units:
                val_units = self.get_value_units_dict_tex()
            if curves_is_str and (not legend_title):
                if val_units[curves]:
                    legend_title = r"$\begin{array}{c}"+rf"\mathrm{{{curves}}} \\"+r"\left["+val_units[curves]+r"\right]"+r"\end{array}$"
                else:
                    legend_title = curves

            if not xlabel:
                if val_units[x_axis]:
                    xlabel = f"{x_axis} " + r"$\left[" + val_units[x_axis] + r"\right]$"
                else:
                    xlabel = f"{x_axis}"
            
            if not ylabel:
                if attrib_name:
                    if attrib_unit:
                        ylabel = f"{attrib_name} " + r"$\left["+attrib_unit+r"\right]$"
                    else:
                        ylabel = attrib_name
                else:
                    ylabel = None

        else:
            plt.rcParams.update({
                "font.size" : global_fontsize
            })
            if not val_units:
                val_units = self.get_value_units_dict()
            if curves_is_str and (not legend_title):
                if val_units[curves]:
                    legend_title = f"{curves}\n [{val_units[curves]}]"
                else:
                    legend_title = curves
            
            if not xlabel:
                if val_units[x_axis]:
                    xlabel = f"{x_axis} " + "[" + val_units[x_axis] + "]"
                else:
                    xlabel = f"{x_axis}"
            
            if not ylabel:
                if attrib_name:
                    if attrib_unit:
                        ylabel = f"{attrib_name} [{attrib_unit}]"
                    else:
                        ylabel = attrib_name
                else:
                    ylabel = None
                    
        if x_axis_vals is None:
            x_axis_vals = self.available_params[x_axis]
        
        fig = plt.figure(figsize=plot_figsize)

        curves_to_plot = {}
        for curve_name, curve_params in curves_dict.items():
            attrib_vals = []
            x_axis_vals_plot = []
            for x_axis_val in x_axis_vals:
                x_axis_val = round(x_axis_val, round_xvals)
                curve_params[x_axis] = x_axis_val
                sim_params = common_params | curve_params
                try:
                    sim = self.get_simulation(sim_params, is_internal = True)
                except Exception as e:
                    raise ValueError(f"Parameters must uniquely define simulations. common_params has current value {common_params}. Ensure that all parameters not included in 'x_axis' and 'curves' are specified.") from e
                if sim:
                    attrib_vals.append(result_func(sim))
                    x_axis_vals_plot.append(x_axis_val)
            curves_to_plot[curve_name] = {
                "x"     : x_axis_vals_plot,
                "y"     : attrib_vals,
                "avg"   : np.mean(attrib_vals)
            }
        if curve_colors is None:
            if curves_is_str and (isinstance(curve_vals[0],float) or isinstance(curve_vals[0],int)):
                norm = mcolors.Normalize(vmin=min(curve_vals) + cmap_min_offset, vmax=max(curve_vals) + cmap_max_offset)
                for curve_name, data in curves_to_plot.items():
                    data["color"] = cmap(norm(curve_name))
            else:
                all_avgs = [d["avg"] for d in curves_to_plot.values()]
                norm = mcolors.Normalize(vmin=min(all_avgs) + cmap_min_offset, vmax=max(all_avgs) + cmap_max_offset)
                for curve_name, data in curves_to_plot.items():
                    data["color"] = cmap(norm(data["avg"]))
                curves_to_plot = dict(sorted(curves_to_plot.items(), key=lambda item: item[1]['avg'], reverse=True))
        else:
            for curve_name, data in curves_to_plot.items():
                data["color"] = curve_colors[curve_name]
        # curves_to_plot = self._get_curves_with_colors_(curves_to_plot,curves_is_str, curve_vals=curve_vals, curve_colors=curve_colors, 
        #                                               cmap_max_offset=cmap_max_offset, cmap_min_offset=cmap_min_offset)

        for curve_name, curve_data in curves_to_plot.items():
            x = curve_data["x"]
            y = curve_data["y"]
            plt_color = curve_data["color"]
            if len(x) > 0:
                if use_interp:
                    pchip = PchipInterpolator(x, y)
                    xx = np.linspace(min(x), max(x), interp_linspace_size)
                    yy = pchip(xx)
                    if show_datapoints:
                        plt.plot(x,y,".",lw = lw+1, color = plt_color)
                    plt.plot(xx,yy, lw = lw,label=f"{curve_name}", color = plt_color)
                else:
                    plt.plot(x,y,lw = lw,label=f"{curve_name}", color = plt_color)
                    if show_datapoints:
                        plt.plot(x,y,".",lw= lw + 1, color = plt_color)
        
        if title:
            plt.title(title, pad=title_pad, fontsize=title_fontsize)
        if xlabel:
            plt.xlabel(xlabel, labelpad=xlabel_pad, fontsize=xlabel_fontsize)
        if ylabel:
            plt.ylabel(ylabel, labelpad=ylabel_pad, fontsize=ylabel_fontsize)
        if show_legend:
            plt.legend(bbox_to_anchor=legend_bbox_to_anchor, loc=legend_loc).set_title(legend_title)
        
        plt.tight_layout()
        plt.xlim(left = left_lim, right = right_lim)
        plt.ylim(bottom = bottom_lim, top = top_lim)
        if grid == "major":
            plt.grid(which='major', linestyle=grid_major_linestyle, linewidth=grid_major_linewidth)
        if grid == "minor":
            plt.minorticks_on()
            plt.grid(which='major', linestyle=grid_major_linestyle, linewidth=grid_major_linewidth)
            plt.grid(which='minor', linestyle=grid_minor_linestyle, linewidth=grid_minor_linewidth)

        if returnfig:
            return fig
        if savefig:
            if output_filename:
                plt.savefig(os.path.join(save_folder,output_filename), dpi = dpi)
            else:
                plt.savefig(os.path.join(save_folder,f"plot_{time()}.png"), dpi = dpi)
        if show:
            plt.show()
        
        plt.close(fig)
    
    def plot_heat_load(self, x_axis: str, curves: Union[str, dict], common_params: dict = {}, x_axis_vals: list = None, curve_vals: list = None,
                       T_rev: float = 88.9e-6, unit: str = "mW", usetex: bool = True, global_fontsize: float = 18, curve_colors: dict = None,
                       use_interp: bool = True, interp_linspace_size: int = 300, show_datapoints: bool = True, lw: float = 2,
                       plot_figsize : tuple = (10,5), cmap = plt.cm.magma, cmap_min_offset: float = 0, cmap_max_offset: float = 0,
                       val_units: dict = None, show_legend: bool = True, legend_title: str = None, legend_bbox_to_anchor: tuple = (1.04, 0.5), legend_loc: str = "center left",
                       left_lim: float = None, right_lim: float = None, bottom_lim: float = 0, top_lim: float = None,
                       title: str = None, title_pad: float = 20, title_fontsize: float = 20,
                       xlabel: str = None, xlabel_pad: float = 10, xlabel_fontsize: float = 20,
                       ylabel: str = None, ylabel_pad: float = 10, ylabel_fontsize: float = 20,
                       grid: str = "minor", grid_major_linestyle: str = "-", grid_major_linewidth: float = 0.75,
                       grid_minor_linestyle: str = ":", grid_minor_linewidth: float = 0.5,
                       savefig: bool = False, output_filename: str = "heat_load.png", dpi: int = 300, show: bool = True, save_folder: str = "./",
                       returnfig: bool = False, round_xvals: int = 5, round_curvevals: int = 5
                       ):

        dispunit = {
            "mW" : "mW",
            "W"  : "W",
            "eV" : "eV/s"
        }
        
        if not title:
            if isinstance(curves,str):
                title = f"Heat load per bunch and unit length as a function of {x_axis} and {curves}"
            else:
                title = f"Heat load per bunch and unit length as a function of {x_axis}"
        if not ylabel:
            ylabel = f"Heat load [{dispunit[unit]}/m/bunch]"
        
        def heat_load_func(sim, T_rev = T_rev, unit = unit):
            return sim.calculate_heat_load_per_bunch(T_rev = T_rev, unit = unit)

        self.plot_simulation_result_vs_attrib(heat_load_func, x_axis, curves, common_params = common_params , x_axis_vals = x_axis_vals, curve_vals = curve_vals,
                       usetex = usetex, global_fontsize = global_fontsize, curve_colors = curve_colors,
                       use_interp = use_interp, interp_linspace_size = interp_linspace_size, show_datapoints = show_datapoints, lw = lw,
                       plot_figsize = plot_figsize, cmap = cmap, cmap_min_offset = cmap_min_offset, cmap_max_offset = cmap_max_offset,
                       val_units = val_units, show_legend = show_legend, legend_title = legend_title, legend_bbox_to_anchor = legend_bbox_to_anchor, legend_loc = legend_loc,
                       left_lim = left_lim, right_lim = right_lim, bottom_lim = bottom_lim, top_lim = top_lim,
                       title = title, title_pad = title_pad, title_fontsize = title_fontsize,
                       xlabel = xlabel, xlabel_pad = xlabel_pad, xlabel_fontsize = xlabel_fontsize,
                       ylabel = ylabel, ylabel_pad = ylabel_pad, ylabel_fontsize = ylabel_fontsize,
                       grid = grid, grid_major_linestyle = grid_major_linestyle, grid_major_linewidth = grid_major_linewidth,
                       grid_minor_linestyle = grid_minor_linestyle, grid_minor_linewidth = grid_minor_linewidth,
                       savefig = savefig, output_filename = output_filename, dpi = dpi, show = show, save_folder = save_folder,
                       returnfig = returnfig, round_xvals = round_xvals, round_curvevals = round_curvevals
                       )
    
    def plot_max_cen_density(self, x_axis: str, curves: Union[str, dict], common_params: dict = {}, x_axis_vals: list = None, curve_vals: list = None,
                            usetex: bool = True, global_fontsize: float = 15, curve_colors: dict = None,
                            use_interp: bool = True, interp_linspace_size: int = 300, show_datapoints: bool = True, lw: float = 2,
                            plot_figsize : tuple = (8,5), cmap = plt.cm.magma, cmap_min_offset: float = 0, cmap_max_offset: float = 0,
                            val_units: dict = None, show_legend: bool = True, legend_title: str = None, legend_bbox_to_anchor: tuple = (1.04, 0.5), legend_loc: str = "center left",
                            left_lim: float = None, right_lim: float = None, bottom_lim: float = 10**9, top_lim: float = None,
                            title: str = None, title_pad: float = 50, title_fontsize: float = 20,
                            xlabel: str = None, xlabel_pad: float = 10, xlabel_fontsize: float = 20,
                            ylabel: str = None, ylabel_pad: float = 15, ylabel_fontsize: float = 20,
                            grid: str = "minor", grid_major_linestyle: str = "-", grid_major_linewidth: float = 0.75,
                            grid_minor_linestyle: str = ":", grid_minor_linewidth: float = 0.5,
                            savefig: bool = False, output_filename: str = "cen_density.png", dpi: int = 300, show: bool = True, save_folder: str = "./",
                            returnfig: bool = False, round_xvals: int = 5, round_curvevals: int = 5,
                            mode: str = "last_idx", hist_n_last_indices: int = 6, hist_bin_height: float = 1.800000e-02):
        if usetex:            
            if not ylabel:
                ylabel = r"Central Electron Density [$ \rm m^{-3}$]"

        else:
            if not ylabel:
                ylabel = f"Central Electron Density [m^-3]"

        if not title:
            title = f"Central Electron Density as a function of {x_axis} and {curves}"            

        if mode == "max_cen":
            def calculate_max_cen_density(sim):
                return max(sim.cen_density)
        elif mode == "x_hist":
            def calculate_max_cen_density(sim, n_indices = hist_n_last_indices, bin_height = hist_bin_height):
                x_hist, nel_hist = sim.get_horizontal_nel_hist_i_passage(n_indices = n_indices, bin_height = bin_height)
                center_idx = np.squeeze(np.where(x_hist == 0)[0])
                dens_around_center = [nel_hist[center_idx - 1],nel_hist[center_idx],nel_hist[center_idx + 1]]
                return max(dens_around_center)
        elif mode == "last_idx":
            def calculate_max_cen_density(sim):
                idx1, idx2 = sim.get_i_train_last_bunch_start()
                # Debugging
                # plt.plot(sim.t,sim.cen_density, color = 'b')
                # plt.plot(sim.t[idx1:idx2],sim.cen_density[idx1:idx2],lw = 3, color = 'r')
                # plt.show()
                return np.mean(sim.cen_density[idx1:idx2])
        else:
            raise ValueError("Available modes for calculating central electron density are 'last_idx', 'max_cen' and 'x_hist'")

        fig = self.plot_simulation_result_vs_attrib(calculate_max_cen_density, x_axis, curves, common_params = common_params , x_axis_vals = x_axis_vals, curve_vals = curve_vals,
                       usetex = usetex, global_fontsize = global_fontsize, curve_colors = curve_colors,
                       use_interp = use_interp, interp_linspace_size = interp_linspace_size, show_datapoints = show_datapoints, lw = lw,
                       plot_figsize = plot_figsize, cmap = cmap, cmap_min_offset = cmap_min_offset, cmap_max_offset = cmap_max_offset,
                       val_units = val_units, show_legend = show_legend, legend_title = legend_title, legend_bbox_to_anchor = legend_bbox_to_anchor, legend_loc = legend_loc,
                       left_lim = left_lim, right_lim = right_lim, bottom_lim = bottom_lim, top_lim = top_lim,
                       title = title, title_pad = title_pad, title_fontsize = title_fontsize,
                       xlabel = xlabel, xlabel_pad = xlabel_pad, xlabel_fontsize = xlabel_fontsize,
                       ylabel = ylabel, ylabel_pad = ylabel_pad, ylabel_fontsize = ylabel_fontsize,
                       grid = grid, grid_major_linestyle = grid_major_linestyle, grid_major_linewidth = grid_major_linewidth,
                       grid_minor_linestyle = grid_minor_linestyle, grid_minor_linewidth = grid_minor_linewidth,
                       savefig = False, output_filename = output_filename, dpi = dpi, show = False, save_folder = save_folder,
                       returnfig = True, round_xvals = round_xvals, round_curvevals = round_curvevals
                       )
        
        plt.semilogy()
        plt.tight_layout()
        # Enable color indicating instability after threshold
        # ax = fig.axes[0]
        # ax.axhspan(5*10**11,ax.get_ylim()[1],color='red',alpha=0.2)
        if returnfig:
            return fig
        if savefig:
            plt.savefig(os.path.join(save_folder,output_filename), dpi = dpi)
        if show:
            plt.show()
        
        plt.close(fig)
        
    def plot_simulation_attribs(self, x_axis_attrib: Union[str, Callable[[Any], list[float]]] ,y_axis_attrib : Union[str, Callable[[Any], list[float]]], curves : Union[str, dict] = None, 
                                common_params: dict = {}, curve_vals: list = None, usetex: bool = True, global_fontsize: float = 18, curve_colors: dict = None,
                                use_interp: bool = True, interp_linspace_size: int = 300, show_datapoints: bool = True, lw: float = 2,
                                plot_figsize : tuple = (10,5), cmap = plt.cm.magma, cmap_min_offset: float = 0, cmap_max_offset: float = 0,
                                val_units: dict = None, show_legend: bool = True, legend_title: str = None, legend_bbox_to_anchor: tuple = (1.04, 0.5), legend_loc: str = "center left",
                                left_lim: float = None, right_lim: float = None, bottom_lim: float = None, top_lim: float = None,
                                title: str = None, title_pad: float = 20, title_fontsize: float = 20,
                                xlabel: str = None, xlabel_pad: float = 10, xlabel_fontsize: float = 20,
                                ylabel: str = None, ylabel_pad: float = 10, ylabel_fontsize: float = 20,
                                grid: str = "minor", grid_major_linestyle: str = "-", grid_major_linewidth: float = 0.75,
                                grid_minor_linestyle: str = ":", grid_minor_linewidth: float = 0.5,
                                savefig: bool = False, output_filename: str = None, dpi: int = 300, show: bool = True, save_folder: str = "./",
                                returnfig: bool = False, round_curvevals: int = 5
                                ):
        '''if isinstance(x, str):
            x_val = getattr(sim, x)
        elif callable(x):
            x_val = x(sim)
        else:
            raise TypeError("x must be a string or a callable")'''
        mpl.rcParams.update(mpl.rcParamsDefault)

        def get_attrib(x: Union[str, Callable[[Any], list[float]]], sim):
            if isinstance(x, str):
                return getattr(sim, x)
            elif callable(x):
                return x(sim)
            else:
                raise TypeError("x must be a string or a callable")
        
        def get_attrib_name(x: Union[str, Callable[[Any], list[float]]]):
            if isinstance(x, str):
                return x
            elif callable(x):
                return x.__name__
            else:
                raise TypeError("x must be a string or a callable")
        if curves is not None:
            curves_dict = {}
            if isinstance(curves,str):
                if curves.endswith(".yaml"):
                    curves_dict = self.read_yaml_to_dict(curves)
                    curves_is_str = False
                else:
                    if curves in self.sims_per_parameter.keys():
                        if not (curves in self.available_params.keys()):
                            ValueError(f"'curves' must be iterable. (In available parameters for plotting: {self.print_available_params_str})")
                    else:
                        ValueError(f"Parameter {curves} not in list of tracked parameters.")
                    if curve_vals is None:
                        curve_vals = self.available_params[curves]
                        curves_dict = {v: {curves : v} for v in curve_vals}
                    else:
                        curves_dict = {round(v, round_curvevals): {curves : round(v, round_curvevals)} for v in curve_vals}
                    curves_is_str = True
            elif isinstance(curves,dict):
                if curve_vals:
                    raise ValueError("If curve_vals is specified, 'curves' must uniquely define a parameter that can be iterable")
                curves_dict = curves
                curves_is_str = False
            else:
                raise ValueError("'curves' must either define a specific parameter as 'str' or a group of parameters as 'dict'")
        else:
            curves_is_str = False
        if usetex:
            plt.rcParams.update({
                "text.usetex": True,        # Use LaTeX to render all text
                "font.family": "serif",     # Use serif fonts (Computer Modern is default)
                "font.serif": ["Computer Modern"],  # Optionally specify which serif font
                "font.size" : global_fontsize
            })

            if not val_units:
                val_units = self.get_value_units_dict_tex()
            if not legend_title and curves_is_str:
                if val_units[curves]:
                    legend_title = r"$\begin{array}{c}"+rf"\mathrm{{{curves}}} \\"+r"\left["+val_units[curves]+r"\right]"+r"\end{array}$"
                else:
                    legend_title = curves
        else:
            plt.rcParams.update({
                "font.size" : global_fontsize
            })
            if not val_units:
                val_units = self.get_value_units_dict()
            if not legend_title and curves_is_str:
                if val_units[curves]:
                    legend_title = f"{curves}\n [{val_units[curves]}]"
                else:
                    legend_title = curves

        if not xlabel:
            xlabel = get_attrib_name(x_axis_attrib)
        if not ylabel:
            ylabel = get_attrib_name(y_axis_attrib)

        if curves is not None:
            fig = plt.figure(figsize=plot_figsize)
            curves_to_plot = {}
            for curve_name, curve_params in curves_dict.items():
                sim_params = common_params | curve_params
                try:
                    sim = self.get_simulation(sim_params, is_internal = True)
                except Exception as e:
                    raise ValueError(f"Parameters must uniquely define simulations. common_params has current value {common_params}. Ensure that all parameters not included in 'x_axis' and 'curves' are specified.") from e
                if sim:
                    x_axis_vals = get_attrib(x_axis_attrib, sim)
                    y_axis_vals = get_attrib(y_axis_attrib, sim)
                    curves_to_plot[curve_name] = {
                        "x"     : x_axis_vals,
                        "y"     : y_axis_vals,
                        "avg"   : np.mean(y_axis_vals)
                    }
            if curve_colors is None:
                if curves_is_str and (isinstance(curve_vals[0],float) or isinstance(curve_vals[0],int)):
                    norm = mcolors.Normalize(vmin=min(curve_vals) + cmap_min_offset, vmax=max(curve_vals) + cmap_max_offset)
                    for curve_name, data in curves_to_plot.items():
                        data["color"] = cmap(norm(curve_name))
                else:
                    all_avgs = [d["avg"] for d in curves_to_plot.values()]
                    norm = mcolors.Normalize(vmin=min(all_avgs) + cmap_min_offset, vmax=max(all_avgs) + cmap_max_offset)
                    for curve_name, data in curves_to_plot.items():
                        data["color"] = cmap(norm(data["avg"]))
                    curves_to_plot = dict(sorted(curves_to_plot.items(), key=lambda item: item[1]['avg'], reverse=True))
            else:
                for curve_name, data in curves_to_plot.items():
                    data["color"] = curve_colors[curve_name]

            for curve_name, curve_data in curves_to_plot.items():
                x = curve_data["x"]
                y = curve_data["y"]
                plt_color = curve_data["color"]
                if use_interp:
                    pchip = PchipInterpolator(x, y)
                    xx = np.linspace(min(x), max(x), interp_linspace_size)
                    yy = pchip(xx)
                    if show_datapoints:
                        plt.plot(x,y,".",lw = lw+1, color = plt_color)
                    plt.plot(xx,yy, lw = lw,label=f"{curve_name}", color = plt_color)
                else:
                    plt.plot(x,y,lw = lw,label=f"{curve_name}", color = plt_color)
                    if show_datapoints:
                        plt.plot(x,y,".",lw= lw + 1, color = plt_color)
        else:
            try:
                sim = self.get_simulation(common_params, is_internal = True)
            except Exception as e:
                raise ValueError(f"Parameters must uniquely define simulations. common_params has current value {common_params}. Ensure that all parameters not included in 'x_axis' and 'curves' are specified.") from e
            if sim:
                if use_interp:
                    x = get_attrib(x_axis_attrib, sim)
                    y = get_attrib(y_axis_attrib, sim)
                    pchip = PchipInterpolator(x, y)
                    xx = np.linspace(min(x), max(x), interp_linspace_size)
                    yy = pchip(xx)
                    if show_datapoints:
                        plt.plot(x,y,".",lw = lw+1, color = "k")
                    plt.plot(xx,yy, lw = lw, color = "k")
                else:
                    plt.plot(x,y,lw = lw, color = "k")
                    if show_datapoints:
                        plt.plot(x,y,".",lw= lw + 1, color = "k")

        if title:
            plt.title(title, pad=title_pad, fontsize=title_fontsize)
        if xlabel:
            plt.xlabel(xlabel, labelpad=xlabel_pad, fontsize=xlabel_fontsize)
        if ylabel:
            plt.ylabel(ylabel, labelpad=ylabel_pad, fontsize=ylabel_fontsize)
        if show_legend and curves is not None:
            plt.legend(bbox_to_anchor=legend_bbox_to_anchor, loc=legend_loc).set_title(legend_title)
        
        plt.tight_layout()
        plt.xlim(left = left_lim, right = right_lim)
        plt.ylim(bottom = bottom_lim, top = top_lim)

        if grid == "major":
            plt.grid(which='major', linestyle=grid_major_linestyle, linewidth=grid_major_linewidth)
        if grid == "minor":
            plt.minorticks_on()
            plt.grid(which='major', linestyle=grid_major_linestyle, linewidth=grid_major_linewidth)
            plt.grid(which='minor', linestyle=grid_minor_linestyle, linewidth=grid_minor_linewidth)

        if returnfig:
            return fig
        if savefig:
            if output_filename:
                plt.savefig(os.path.join(save_folder,output_filename), dpi = dpi)
            else:
                plt.savefig(os.path.join(save_folder,f"plot_{time()}.png"), dpi = dpi)
        if show:
            plt.show()
        
        plt.close(fig)
    
    def plot_buildup(self, curves : Union[str, dict] = None, common_params: dict = {}, 
                        curve_vals: list = None, usetex: bool = True, global_fontsize: float = 18, curve_colors: dict = None,
                        use_interp: bool = True, interp_linspace_size: int = 300, show_datapoints: bool = True, lw: float = 2,
                        plot_figsize : tuple = (10,5), cmap = plt.cm.magma, cmap_min_offset: float = 0, cmap_max_offset: float = 0,
                        val_units: dict = None, show_legend: bool = True, legend_title: str = None, legend_bbox_to_anchor: tuple = (1.04, 0.5), legend_loc: str = "center left",
                        left_lim: float = None, right_lim: float = None, bottom_lim: float = None, top_lim: float = None,
                        title: str = None, title_pad: float = 20, title_fontsize: float = 20,
                        xlabel: str = None, xlabel_pad: float = 10, xlabel_fontsize: float = 20,
                        ylabel: str = None, ylabel_pad: float = 10, ylabel_fontsize: float = 20,
                        grid: str = "minor", grid_major_linestyle: str = "-", grid_major_linewidth: float = 0.75,
                        grid_minor_linestyle: str = ":", grid_minor_linewidth: float = 0.5,
                        savefig: bool = False, output_filename: str = None, dpi: int = 300, show: bool = True, save_folder: str = "./",
                        returnfig: bool = False, round_curvevals: int = 5
                        ):
        def get_y(sim):
            return np.sum(sim.nel_hist,axis = 1)/sim.chamber_area
        
        if usetex:
            if not xlabel:
                xlabel = r"Time [$\mathrm{s}$]"
            if not ylabel:
                ylabel = r"Electron Density [$\mathrm{m}^{-3}$]"
        else:
            if not xlabel:
                xlabel = "Time [s]"
            if not ylabel:
                ylabel = "Electron Density [m^-3]"
        
        if not title:
            if curves is not None:
                title = f"Electron buildup in chamber as a function of {curves}"
            else:
                title = "Electron buildup in chamber"
        self.plot_simulation_attribs("t_hist" ,get_y, curves = curves, common_params = common_params,
                                curve_vals = curve_vals, usetex = usetex, global_fontsize = global_fontsize, curve_colors = curve_colors,
                                use_interp = use_interp, interp_linspace_size = interp_linspace_size, show_datapoints = show_datapoints, lw = lw,
                                plot_figsize = plot_figsize, cmap = cmap, cmap_min_offset = cmap_min_offset, cmap_max_offset = cmap_max_offset,
                                val_units = val_units, show_legend = show_legend, legend_title = legend_title, legend_bbox_to_anchor = legend_bbox_to_anchor, legend_loc = legend_loc,
                                left_lim = left_lim, right_lim = right_lim, bottom_lim = bottom_lim, top_lim = top_lim,
                                title = title, title_pad = title_pad, title_fontsize = title_fontsize,
                                xlabel = xlabel, xlabel_pad = xlabel_pad, xlabel_fontsize = xlabel_fontsize,
                                ylabel = ylabel, ylabel_pad = ylabel_pad, ylabel_fontsize = ylabel_fontsize,
                                grid = grid, grid_major_linestyle = grid_major_linestyle, grid_major_linewidth = grid_major_linewidth,
                                grid_minor_linestyle = grid_minor_linestyle, grid_minor_linewidth = grid_minor_linewidth,
                                savefig = savefig, output_filename = output_filename, dpi = dpi, show = show, save_folder = save_folder,
                                returnfig = returnfig, round_curvevals = round_curvevals)
    
    def plot_horizontal_electron_hist(self, curves : Union[str, dict] = None, common_params: dict = {}, 
                        curve_vals: list = None, usetex: bool = True, global_fontsize: float = 18, curve_colors: dict = None,
                        use_interp: bool = True, interp_linspace_size: int = 300, show_datapoints: bool = True, lw: float = 2,
                        plot_figsize : tuple = (10,5), cmap = plt.cm.magma, cmap_min_offset: float = 0, cmap_max_offset: float = 0,
                        val_units: dict = None, show_legend: bool = True, legend_title: str = None, legend_bbox_to_anchor: tuple = (1.04, 0.5), legend_loc: str = "center left",
                        left_lim: float = None, right_lim: float = None, bottom_lim: float = 10**9, top_lim: float = None,
                        title: str = None, title_pad: float = 20, title_fontsize: float = 20,
                        xlabel: str = None, xlabel_pad: float = 10, xlabel_fontsize: float = 20,
                        ylabel: str = None, ylabel_pad: float = 10, ylabel_fontsize: float = 20,
                        grid: str = "minor", grid_major_linestyle: str = "-", grid_major_linewidth: float = 0.75,
                        grid_minor_linestyle: str = ":", grid_minor_linewidth: float = 0.5,
                        savefig: bool = False, output_filename: str = None, dpi: int = 300, show: bool = True, save_folder: str = "./",
                        returnfig: bool = False, round_curvevals: int = 5,
                        train_num = -1, n_last_indices_to_avg: int = 6, hist_bin_height: float = 1.800000e-02):
        
        def get_y(sim, train_num = train_num, n_indices = n_last_indices_to_avg, bin_height = hist_bin_height):
            return sim.get_horizontal_nel_hist_i_passage(train_num = train_num, n_indices = n_indices, bin_height = bin_height)[1]
        
        if usetex:
            if not xlabel:
                xlabel = r"$x_{\rm pos}$ [$\mathrm{m}$]"
            if not ylabel:
                ylabel = r"Electron Density [$\mathrm{m}^{-3}$]"
        else:
            if not xlabel:
                xlabel = "x [m]"
            if not ylabel:
                ylabel = "Electron Density [m^-3]"
        
        if not title:
            if curves is not None:
                title = f"Horizontal electron density in chamber as a function of {curves}"
            else:
                title = "Horizontal electron density in chamber"

        fig = self.plot_simulation_attribs("xg_hist" ,get_y, curves = curves, common_params = common_params, 
                                curve_vals = curve_vals, usetex = usetex, global_fontsize = global_fontsize, curve_colors = curve_colors,
                                use_interp = use_interp, interp_linspace_size = interp_linspace_size, show_datapoints = show_datapoints, lw = lw,
                                plot_figsize = plot_figsize, cmap = cmap, cmap_min_offset = cmap_min_offset, cmap_max_offset = cmap_max_offset,
                                val_units = val_units, show_legend = show_legend, legend_title = legend_title, legend_bbox_to_anchor = legend_bbox_to_anchor, legend_loc = legend_loc,
                                left_lim = left_lim, right_lim = right_lim, bottom_lim = bottom_lim, top_lim = top_lim,
                                title = title, title_pad = title_pad, title_fontsize = title_fontsize,
                                xlabel = xlabel, xlabel_pad = xlabel_pad, xlabel_fontsize = xlabel_fontsize,
                                ylabel = ylabel, ylabel_pad = ylabel_pad, ylabel_fontsize = ylabel_fontsize,
                                grid = grid, grid_major_linestyle = grid_major_linestyle, grid_major_linewidth = grid_major_linewidth,
                                grid_minor_linestyle = grid_minor_linestyle, grid_minor_linewidth = grid_minor_linewidth,
                                savefig = False, output_filename = output_filename, dpi = dpi, show = False, save_folder = save_folder,
                                returnfig = True, round_curvevals = round_curvevals)
        plt.semilogy()
        plt.tight_layout()
        
        if returnfig:
            return fig
        if savefig:
            if output_filename:
                plt.savefig(os.path.join(save_folder,output_filename), dpi = dpi)
            else:
                plt.savefig(os.path.join(save_folder,f"plot_{time()}.png"), dpi = dpi)
        if show:
            plt.show()
        
        plt.close(fig)
        
    def plot_half_cell_heat_load(self, magnet_config: Union[str, dict], x_axis: str, curves: Union[str, dict], common_params: dict = {}, 
                       curve_colors: dict = None, attrib_name: str = None, attrib_unit: str = None, x_axis_vals: list = None, curve_vals: list = None, 
                       T_rev: float = 88.9e-6, unit: str = "mW", usetex: bool = True, global_fontsize: float = 18,
                       use_interp: bool = True, interp_linspace_size: int = 300, show_datapoints: bool = True, lw: float = 2,
                       plot_figsize : tuple = (10,5), cmap = plt.cm.magma, cmap_min_offset: float = 0, cmap_max_offset: float = 0 ,
                       val_units: dict = None, show_legend: bool = True, legend_title: str = None, legend_bbox_to_anchor: tuple = (1.04, 0.5), legend_loc: str = "center left",
                       left_lim: float = None, right_lim: float = None, bottom_lim: float = 0, top_lim: float = None,
                       title: str = None, title_pad: float = 20, title_fontsize: float = 20,
                       xlabel: str = None, xlabel_pad: float = 10, xlabel_fontsize: float = 20,
                       ylabel: str = None, ylabel_pad: float = 10, ylabel_fontsize: float = 20,
                       grid: str = "minor", grid_major_linestyle: str = "-", grid_major_linewidth: float = 0.75,
                       grid_minor_linestyle: str = ":", grid_minor_linewidth: float = 0.5,
                       savefig: bool = False, output_filename: str = None, dpi: int = 300, show: bool = True, save_folder: str = "./",
                       returnfig: bool = False, round_xvals: int = 5, round_curvevals: int = 5
                       ):
        mpl.rcParams.update(mpl.rcParamsDefault)
        if not (isinstance(self.available_params[x_axis][0], float) or isinstance(self.available_params[x_axis][0], int)):
            raise ValueError("Values on x_axis cannot be non numeric")
        
        
        if isinstance(magnet_config,str):
            if curves.endswith(".yaml"):
                curves_dict = self.read_yaml_to_dict(curves)
            else:
                ValueError("'magnet_config' must either be a yaml file or dict")
        elif isinstance(magnet_config, dict):
            pass
        else:
            raise ValueError("'magnet_config' must either be a yaml file or dict")
        
        curves_dict = {}
        if isinstance(curves,str):
            if curves.endswith(".yaml"):
                curves_dict = self.read_yaml_to_dict(curves)
                curves_is_str = False
            else:
                if curve_vals is None:
                    curve_vals = self.available_params[curves]
                    curves_dict = {v: {curves : v} for v in curve_vals}
                else:
                    curves_dict = {round(v, round_curvevals): {curves : round(v, round_curvevals)} for v in curve_vals}
                curves_is_str = True
        elif isinstance(curves,dict):
            if curve_vals:
                raise ValueError("If curve_vals is specified, 'curves' must uniquely define a parameter that can be iterable")
            curves_dict = curves
            curves_is_str = False
        else:
            raise ValueError("'curves' must either define a specific parameter as 'str' or a group of parameters as 'dict'")

        dispunit = {
            "mW" : "mW",
            "W"  : "W",
            "eV" : "eV/s"
        }
        
        if not ylabel:
            ylabel = f"Heat load [{dispunit[unit]}/bunch]"
        if usetex:
            plt.rcParams.update({
                "text.usetex": True,        # Use LaTeX to render all text
                "font.family": "serif",     # Use serif fonts (Computer Modern is default)
                "font.serif": ["Computer Modern"],  # Optionally specify which serif font
                "font.size" : global_fontsize
            })

            if not val_units:
                val_units = self.get_value_units_dict_tex()
            if curves_is_str and (not legend_title):
                if val_units[curves]:
                    legend_title = r"$\begin{array}{c}"+rf"\mathrm{{{curves}}} \\"+r"\left["+val_units[curves]+r"\right]"+r"\end{array}$"
                else:
                    legend_title = curves

            if not xlabel:
                if val_units[x_axis]:
                    xlabel = f"{x_axis} " + r"$\left[" + val_units[x_axis] + r"\right]$"
                else:
                    xlabel = f"{x_axis}"
            
            if not ylabel:
                if attrib_name:
                    if attrib_unit:
                        ylabel = f"{attrib_name} " + r"$\left["+attrib_unit+r"\right]$"
                    else:
                        ylabel = attrib_name
                else:
                    ylabel = None

        else:
            plt.rcParams.update({
                "font.size" : global_fontsize
            })
            if not val_units:
                val_units = self.get_value_units_dict()
            if curves_is_str and (not legend_title):
                if val_units[curves]:
                    legend_title = f"{curves}\n [{val_units[curves]}]"
                else:
                    legend_title = curves
            
            if not xlabel:
                if val_units[x_axis]:
                    xlabel = f"{x_axis} " + "[" + val_units[x_axis] + "]"
                else:
                    xlabel = f"{x_axis}"
            
            if not ylabel:
                if attrib_name:
                    if attrib_unit:
                        ylabel = f"{attrib_name} [{attrib_unit}]"
                    else:
                        ylabel = attrib_name
                else:
                    ylabel = None
        if not title:
            if isinstance(curves,str):
                title = f"Heat load in half cell as a function of {x_axis} and {curves}"
            else:
                title = f"Heat load in half cell as a function of {x_axis}"
        if x_axis_vals is None:
            x_axis_vals = self.available_params[x_axis]
        
        fig = plt.figure(figsize=plot_figsize)

        n_elements_in_cell = len(magnet_config.keys())
        curves_to_plot = {}
        for curve_name, curve_params in curves_dict.items():
            attrib_vals = []
            x_axis_vals_plot = []
            for x_axis_val in x_axis_vals:
                x_axis_val = round(x_axis_val, round_xvals)
                curve_params[x_axis] = x_axis_val
                sim_params = common_params | curve_params
                n_elements = 0
                total_heat_load = 0
                for config in magnet_config.keys():
                    element_params = sim_params.copy()
                    element_params['Magnet Configuration'] = config
                    heat_load_element = 0
                    try:
                        sim = self.get_simulation(element_params, is_internal = True)
                    except Exception as e:
                        raise ValueError(f"Parameters must uniquely define simulations. common_params has current value {common_params}. Ensure that all parameters not included in 'x_axis' and 'curves' are specified.") from e
                    if sim:
                        heat_load_element = sim.calculate_heat_load_per_bunch(T_rev = T_rev, unit = unit)
                        total_heat_load += heat_load_element*magnet_config[config] # 4920*
                        n_elements +=1
                if n_elements == n_elements_in_cell:
                    attrib_vals.append(total_heat_load)
                    x_axis_vals_plot.append(x_axis_val)
            
            curves_to_plot[curve_name] = {
                "x"     : x_axis_vals_plot,
                "y"     : attrib_vals,
                "avg"   : np.mean(attrib_vals)
            }
        if curve_colors is None:
            if curves_is_str and (isinstance(curve_vals[0],float) or isinstance(curve_vals[0],int)):
                norm = mcolors.Normalize(vmin=min(curve_vals) + cmap_min_offset, vmax=max(curve_vals) + cmap_max_offset)
                for curve_name, data in curves_to_plot.items():
                    data["color"] = cmap(norm(curve_name))
            else:
                all_avgs = [d["avg"] for d in curves_to_plot.values()]
                norm = mcolors.Normalize(vmin=min(all_avgs) + cmap_min_offset, vmax=max(all_avgs) + cmap_max_offset)
                for curve_name, data in curves_to_plot.items():
                    data["color"] = cmap(norm(data["avg"]))
                curves_to_plot = dict(sorted(curves_to_plot.items(), key=lambda item: item[1]['avg'], reverse=True))
        else:
            for curve_name, data in curves_to_plot.items():
                data["color"] = curve_colors[curve_name]
                
        for curve_name, curve_data in curves_to_plot.items():
            x = curve_data["x"]
            y = curve_data["y"]
            plt_color = curve_data["color"]
            if len(x) > 0:
                if use_interp:
                    pchip = PchipInterpolator(x, y)
                    xx = np.linspace(min(x), max(x), interp_linspace_size)
                    yy = pchip(xx)
                    if show_datapoints:
                        plt.plot(x,y,".",lw = lw+1, color = plt_color)
                    plt.plot(xx,yy, lw = lw,label=f"{curve_name}", color = plt_color)
                else:
                    plt.plot(x,y,lw = lw,label=f"{curve_name}", color = plt_color)
                    if show_datapoints:
                        plt.plot(x,y,".",lw= lw + 1, color = plt_color)
        
        if title:
            plt.title(title, pad=title_pad, fontsize=title_fontsize)
        if xlabel:
            plt.xlabel(xlabel, labelpad=xlabel_pad, fontsize=xlabel_fontsize)
        if ylabel:
            plt.ylabel(ylabel, labelpad=ylabel_pad, fontsize=ylabel_fontsize)
        if show_legend:
            plt.legend(bbox_to_anchor=legend_bbox_to_anchor, loc=legend_loc).set_title(legend_title)
        
        plt.tight_layout()
        plt.xlim(left = left_lim, right = right_lim)
        plt.ylim(bottom = bottom_lim, top = top_lim)
        if grid == "major":
            plt.grid(which='major', linestyle=grid_major_linestyle, linewidth=grid_major_linewidth)
        if grid == "minor":
            plt.minorticks_on()
            plt.grid(which='major', linestyle=grid_major_linestyle, linewidth=grid_major_linewidth)
            plt.grid(which='minor', linestyle=grid_minor_linestyle, linewidth=grid_minor_linewidth)
        # Shaded area for value exceeding cooling capacity
        # ax = fig.axes[0]
        # ax.axhspan(170,ax.get_ylim()[1],color='red',alpha=0.2)
        if returnfig:
            return fig
        if savefig:
            if output_filename:
                plt.savefig(os.path.join(save_folder,output_filename), dpi = dpi)
            else:
                plt.savefig(os.path.join(save_folder,f"plot_{time()}.png"), dpi = dpi)
        if show:
            plt.show()