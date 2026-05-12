from PyECLOUDplots import PyECLOUDParameterScan
import matplotlib.pyplot as plt
import os

BASE_DIR = "/eos/project/e/ecloud-simulations/ekatrali/LHC_6.8TeV_Arcs_Filling_Pattern_Scan/LHC_6.8TeV_Arcs_3x48_5_trains"
show = False
save = True
skip_existing = False
generate_cen_plots = False
generate_heat_load_plots_sey = True
generate_heat_load_plots_intensity = True
generate_half_cell_load_plots = True
plot_buildup = True
tex_available = False

paramscan = PyECLOUDParameterScan(BASE_DIR)
sim_configs = paramscan.read_yaml_to_dict("./yaml_repo/b_length_3x48.yaml")
manget_conf = paramscan.read_yaml_to_dict("./yaml_repo/magnet_configs.yaml")
# sim_colors = None

length_single_arc_dipole = 14.3
length_single_arc_quadrupole = 3.1

length_half_cell = 53.45

length_dipoles_in_half_cell = length_single_arc_dipole*3
length_quadrupoles_in_half_cell = length_single_arc_quadrupole
length_drifts_in_half_cell = (length_half_cell - length_quadrupoles_in_half_cell - length_dipoles_in_half_cell)

magnets_in_half_cell = {"[8.33/7000*6800]" : length_dipoles_in_half_cell,
                        "[0.0, 182.84444444444443]" : length_quadrupoles_in_half_cell,
                        "[0]" : length_drifts_in_half_cell}

b_lengths = {
    '1.0ns': '1.00000e-09/4.*299792458.',
    '1.1ns': '1.10000e-09/4.*299792458.', 
    '1.2ns': '1.20000e-09/4.*299792458.', 
    '1.3ns': '1.30000e-09/4.*299792458.', 
    '1.4ns': '1.40000e-09/4.*299792458.'
}

sim_colors = {
    '1.0ns': '#1f77b4',
    '1.1ns': '#ff7f0e',
    '1.2ns': '#2ca02c',
    '1.3ns': '#d62728',
    '1.4ns': '#9467bd'
}

legend_title = r"$4\sigma$ Bunch Length"

if generate_cen_plots:
    for sey in paramscan.get_param_values("SEY"):
        for surface_curve in ["CuO", "Cu2O"]:
            for photoemission_stat in ["Conditioned", "Unconditioned"]:
                sims_for_this = {}
                for b_length in b_lengths.keys():
                    sims_for_this[b_length] = { 'Bunch Length': b_lengths[b_length] } | sim_configs[f"{surface_curve} {photoemission_stat} [3x48]"]
                for magnet_config in manget_conf.keys():
                    output_dir = os.path.join(BASE_DIR, f"analysis/central_density/{magnet_config}/{surface_curve}/{photoemission_stat}")
                    os.makedirs(output_dir,exist_ok = True)
                    output_filename = os.path.join(output_dir, f"cen_sey{sey:.2f}_{magnet_config}.png")
                    if skip_existing:
                        if os.path.exists(output_filename):
                            continue
                    
                    title = f"Central Electron Density for {magnet_config} for SEY: {sey} [{surface_curve},{photoemission_stat}, 3x48]"
                    paramscan.plot_max_cen_density("Intensity", sims_for_this, common_params={"SEY":sey, "Magnet Configuration": manget_conf[magnet_config]["Magnet Configuration"]}, 
                                           title=title, cmap = plt.cm.magma, top_lim = 10**14, bottom_lim = 10**8, curve_colors=sim_colors, 
                                           plot_figsize=(10,5), global_fontsize=13, show = show, savefig=save, output_filename=output_filename,
                                                   usetex=tex_available, mode = 'x_hist', legend_title = legend_title)
                    plt.close('all')
                    print(f"\rGenerating central electron density plots: 3x48 {magnet_config} {surface_curve} {photoemission_stat}", end="", flush=True)

heat_load_lims = {"Dipoles": 1.8,
                  "Quadrupoles" : 6.5,
                  "Drift" : 4.5}
if generate_heat_load_plots_sey:
    for intensity in paramscan.get_param_values("Intensity"):
        for surface_curve in ["CuO", "Cu2O"]:
            for photoemission_stat in ["Conditioned", "Unconditioned"]:
                sims_for_this = {}
                for b_length in b_lengths.keys():
                    sims_for_this[b_length] = { 'Bunch Length': b_lengths[b_length] } | sim_configs[f"{surface_curve} {photoemission_stat} [3x48]"]

                for magnet_config in manget_conf.keys():
                    output_dir = os.path.join(BASE_DIR, f"analysis/heat_load_sey/{magnet_config}/{surface_curve}/{photoemission_stat}")
                    os.makedirs(output_dir,exist_ok = True)
                    output_filename = os.path.join(output_dir, f"heatload_intens{intensity:.2f}_{magnet_config}.png")
                    if skip_existing:
                        if os.path.exists(output_filename):
                            continue
                    title = f"Heat load for {magnet_config} for Intensity: {intensity} [{surface_curve},{photoemission_stat}, 3x48]"
                    os.makedirs(output_dir,exist_ok = True)
                    paramscan.plot_heat_load("SEY", sims_for_this, common_params={"Intensity":intensity, "Magnet Configuration": manget_conf[magnet_config]["Magnet Configuration"]}, 
                                             title=title, cmap = plt.cm.magma, curve_colors=sim_colors, top_lim = heat_load_lims[magnet_config],
                                             global_fontsize=13, show = show, savefig=save, output_filename=output_filename,
                                             usetex=tex_available, legend_title = legend_title)
                    plt.close('all')
                    print(f"\rGenerating heat load plots for SEY: Fixed Intensity {intensity} {magnet_config} [{surface_curve},{photoemission_stat}, 3x48]", end="", flush=True)

if generate_heat_load_plots_intensity:
    for sey in paramscan.get_param_values("SEY"):
        for surface_curve in ["CuO", "Cu2O"]:
            for photoemission_stat in ["Conditioned", "Unconditioned"]:
                sims_for_this = {}
                for b_length in b_lengths.keys():
                    sims_for_this[b_length] = { 'Bunch Length': b_lengths[b_length] } | sim_configs[f"{surface_curve} {photoemission_stat} [3x48]"]

                for magnet_config in manget_conf.keys():
                    output_dir = os.path.join(BASE_DIR, f"analysis/heat_load_intensity/{magnet_config}/{surface_curve}/{photoemission_stat}")
                    os.makedirs(output_dir,exist_ok = True)
                    output_filename = os.path.join(output_dir, f"heatload_sey{sey:.2f}_{magnet_config}.png")
                    if skip_existing:
                        if os.path.exists(output_filename):
                            continue
                    title = f"Heat load for {magnet_config} for SEY: {sey} [{surface_curve},{photoemission_stat}, 3x48]"
                    
                    paramscan.plot_heat_load("Intensity", sims_for_this, common_params={"SEY":sey, "Magnet Configuration": manget_conf[magnet_config]["Magnet Configuration"]},
                                             title=title, cmap = plt.cm.magma, curve_colors=sim_colors, top_lim = heat_load_lims[magnet_config],
                                             global_fontsize=13, show = show, savefig=save, output_filename=output_filename,
                                             usetex=tex_available, legend_title = legend_title)
                    plt.close('all')
                    print(f"\rGenerating heat load plots for Intensity: Fixed SEY{sey} {magnet_config} [{surface_curve},{photoemission_stat}, 3x48]", end="", flush=True)

# paramscan.plot_half_cell_heat_load(magnets_in_half_cell, "SEY", "Intensity", common_params=sim_configs["Regular Photoemission"])
if generate_half_cell_load_plots:
    for intensity in paramscan.get_param_values("Intensity"):
        for surface_curve in ["CuO", "Cu2O"]:
            for photoemission_stat in ["Conditioned", "Unconditioned"]:
                sims_for_this = {}
                for b_length in b_lengths.keys():
                    sims_for_this[b_length] = { 'Bunch Length': b_lengths[b_length] } | sim_configs[f"{surface_curve} {photoemission_stat} [3x48]"]

                title = f"Half cell Heat Load for Intensity: {intensity} [{surface_curve},{photoemission_stat}, 3x48]"
                output_dir = os.path.join(BASE_DIR, f"analysis/heat_load/half_cell/{surface_curve}/{photoemission_stat}")
                os.makedirs(output_dir,exist_ok = True)
                output_filename = os.path.join(output_dir, f"halfcell_intens{intensity:.2f}.png")
                if skip_existing:
                    if os.path.exists(output_filename):
                        continue

                paramscan.plot_half_cell_heat_load(magnets_in_half_cell, "SEY", sims_for_this, common_params={"Intensity": intensity}, curve_colors=sim_colors, unit = "W",
                                                   title=title, global_fontsize=13, show = show, savefig=save, output_filename=output_filename,
                                                   usetex=tex_available, ylabel = "Heat load [W/4920 bunches]", legend_title = legend_title)
                plt.close('all')
                print(f"\rGenerating half cell heat load plots for Intensity: {intensity} [{surface_curve},{photoemission_stat}, 3x48]", end="", flush=True)
        
if plot_buildup:
    for magnet_config in manget_conf.keys():
        for intensity in paramscan.get_param_values("Intensity"):
            for sey in [None]: #paramscan.get_param_values("SEY"):
                for surface_curve in ["CuO", "Cu2O"]:
                    for photoemission_stat in ["Conditioned", "Unconditioned"]:
                        sims_for_this = {}
                        for b_length in b_lengths.keys():
                            sims_for_this[b_length] = { 'Bunch Length': b_lengths[b_length] } | sim_configs[f"{surface_curve} {photoemission_stat} [3x48]"]
                            for sim in sims_for_this.keys():
                                title = f"Buildup as a function of SEY [Intensity:{intensity},{sim},{surface_curve},{photoemission_stat}, 3x48]"
                                output_dir = os.path.join(BASE_DIR, f"analysis/buildup/{magnet_config}/{surface_curve}/{photoemission_stat}/{sim}/")
                                os.makedirs(output_dir,exist_ok = True)
                                output_filename = os.path.join(output_dir, f"buildup_intens{intensity:.2f}.png")
                                if skip_existing:
                                    if os.path.exists(output_filename):
                                        continue
                                paramscan.plot_buildup("SEY",common_params=sims_for_this[sim]|{"Intensity":intensity,"Magnet Configuration": manget_conf[magnet_config]["Magnet Configuration"]}, cmap = plt.cm.tab20c,
                                                                        bottom_lim= 10**8, title=title, global_fontsize=13, show_datapoints = False, returnfig = True, output_filename=output_filename,
                                                                        usetex=tex_available)
                                plt.semilogy()
                                plt.tight_layout()
                                if show:
                                    plt.show()
                                if save:
                                    plt.savefig(output_filename+f"buildup_intens{intensity:.2f}.png", dpi = 300)
                                plt.close('all')
