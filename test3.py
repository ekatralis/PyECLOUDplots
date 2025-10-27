from PyECLOUDplots import PyECLOUDParameterScan
import matplotlib.pyplot as plt

show = False
save = True
generate_cen_plots = True
generate_heat_load_plots = False
generate_half_cell_load_plots = False
plot_buildup = False

# paramscan = PyECLOUDParameterScan("C:\\Users\\smbso\\cernbox\\simulations")
paramscan = PyECLOUDParameterScan("C:\\Users\\smbso\\Documents\\CERN\\sim_comparison")
sim_configs = paramscan.read_yaml_to_dict("./yaml_repo_grid/simulations.yaml")
manget_conf = paramscan.read_yaml_to_dict("./yaml_repo_grid/magnet_configs.yaml")
sim_colors = paramscan.read_yaml_to_dict("./yaml_repo_grid/sim_colors.yaml")

length_single_arc_dipole = 14.3
length_single_arc_quadrupole = 3.1

length_half_cell = 53.45

length_dipoles_in_half_cell = length_single_arc_dipole*3
length_quadrupoles_in_half_cell = length_single_arc_quadrupole
length_drifts_in_half_cell = (length_half_cell - length_quadrupoles_in_half_cell - length_dipoles_in_half_cell)

magnets_in_half_cell = {"[8.33/7000*6800]" : length_dipoles_in_half_cell,
                        "[0.0, 182.84444444444443]" : length_quadrupoles_in_half_cell,
                        "[0]" : length_drifts_in_half_cell}

if generate_cen_plots:
    for sey in paramscan.get_param_values("SEY"):
        for magnet_config in manget_conf.keys():
            title = f"Max Central Electron Density for {magnet_config} for SEY: {sey}"
            paramscan.plot_max_cen_density("Intensity", sim_configs, common_params={"SEY":sey, "Magnet Configuration": manget_conf[magnet_config]["Magnet Configuration"]}, 
                                        title=title, cmap = plt.cm.tab20c, top_lim = 10**14, bottom_lim = 10**8, curve_colors=sim_colors, 
                                        plot_figsize=(10,5), global_fontsize=13, show = show, savefig=save, output_filename=f"./plotsnew/cen_sey{sey:.2f}_{magnet_config}.png")
heat_load_lims = {"Dipoles": 1.8,
                  "Quadrupoles" : 6,
                  "No Field" : 3.8}
if generate_heat_load_plots:
    for intensity in paramscan.get_param_values("Intensity"):
        for magnet_config in manget_conf.keys():
            title = f"Heat load for {magnet_config} for Intensity: {intensity}"
            paramscan.plot_heat_load("SEY", sim_configs, common_params={"Intensity":intensity, "Magnet Configuration": manget_conf[magnet_config]["Magnet Configuration"]}, 
                                     title=title, cmap = plt.cm.gist_ncar, curve_colors=sim_colors, top_lim = heat_load_lims[magnet_config],
                                     global_fontsize=13, show = show, savefig=save, output_filename=f"./plotsnew/heatload_intens{intensity:.2f}_{magnet_config}.png")
# paramscan.plot_half_cell_heat_load(magnets_in_half_cell, "SEY", "Intensity", common_params=sim_configs["Regular Photoemission"])
if generate_half_cell_load_plots:
    for intensity in paramscan.get_param_values("Intensity"):
        title = f"Half cell Heat Load for Intensity: {intensity}"
        paramscan.plot_half_cell_heat_load(magnets_in_half_cell, "SEY", sim_configs, common_params={"Intensity": intensity}, curve_colors=sim_colors,
                                           title=title, global_fontsize=13, show = show, savefig=save, output_filename=f"./plotsnew/halfcell_intens{intensity:.2f}.png")
        
if plot_buildup:
    for magnet_config in manget_conf.keys():
        for intensity in paramscan.get_param_values("Intensity"):
            for sey in paramscan.get_param_values("SEY"):
                title = f"Horizontal electron Density for SEY: {sey} and Intensity: {intensity}"
                paramscan.plot_horizontal_electron_hist(sim_configs, common_params={"SEY":sey,"Intensity": intensity, "Magnet Configuration": manget_conf[magnet_config]["Magnet Configuration"]}, curve_colors=sim_colors,
                                                bottom_lim= 10**8, title=title, global_fontsize=13, show = show, savefig=save, output_filename=f"./plotsnew/buildup_{magnet_config}_SEY{sey:.2f}_intens{intensity:.2f}.png")