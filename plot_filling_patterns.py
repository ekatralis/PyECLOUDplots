from PyECLOUDplots import PyECLOUDParameterScan
import matplotlib.pyplot as plt
import os

show = False
save = True
generate_cen_plots = False
generate_heat_load_plots = False
generate_half_cell_load_plots = False
plot_buildup = True
tex_available = False

# paramscan = PyECLOUDParameterScan("C:\\Users\\smbso\\cernbox\\simulations")
paramscan = PyECLOUDParameterScan("/eos/user/e/ekatrali/electron_cloud_buildup_sims/")
sim_configs = paramscan.read_yaml_to_dict("./yaml_repo/added_sims.yaml")
manget_conf = paramscan.read_yaml_to_dict("./yaml_repo/magnet_configs.yaml")
sim_colors = paramscan.read_yaml_to_dict("./yaml_repo/num_bunch_colors.yaml")
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

if generate_cen_plots:
    for sey in paramscan.get_param_values("SEY"):
        for surface_curve in ["CuO", "Cu2O"]:
            for photoemission_stat in ["Conditioned", "Unconditioned"]:
                sims_for_this = {}
                for key in sim_configs.keys():
                    if surface_curve in key and photoemission_stat in key:
                        out_key = key.replace(surface_curve,'')
                        out_key = out_key.replace(' ','')
                        out_key = out_key.replace('[','')
                        out_key = out_key.replace(']','')
                        out_key = out_key.replace(photoemission_stat,'')
                        sims_for_this[out_key] = sim_configs[key]
                for magnet_config in manget_conf.keys():
                    output_filename = f"/eos/user/e/ekatrali/ecloud_plots/central_density/{magnet_config}/{surface_curve}/{photoemission_stat}"
                    title = f"Central Electron Density for {magnet_config} for SEY: {sey} [{surface_curve},{photoemission_stat}]"
                    os.makedirs(output_filename,exist_ok = True)
                    paramscan.plot_max_cen_density("Intensity", sims_for_this, common_params={"SEY":sey, "Magnet Configuration": manget_conf[magnet_config]["Magnet Configuration"]}, 
                                           title=title, cmap = plt.cm.magma, top_lim = 10**14, bottom_lim = 10**8, curve_colors=sim_colors, 
                                           plot_figsize=(10,5), global_fontsize=13, show = show, savefig=save, output_filename=output_filename+f"/cen_sey{sey:.2f}_{magnet_config}.png",
                                                   usetex=tex_available, mode = 'x_hist')
heat_load_lims = {"Dipoles": 1.8,
                  "Quadrupoles" : 6.5,
                  "Drift" : 4.5}
if generate_heat_load_plots:
    for intensity in paramscan.get_param_values("Intensity"):
        for surface_curve in ["CuO", "Cu2O"]:
            for photoemission_stat in ["Conditioned", "Unconditioned"]:
                sims_for_this = {}
                for key in sim_configs.keys():
                    if surface_curve in key and photoemission_stat in key:
                        out_key = key.replace(surface_curve,'')
                        out_key = out_key.replace(' ','')
                        out_key = out_key.replace('[','')
                        out_key = out_key.replace(']','')
                        out_key = out_key.replace(photoemission_stat,'')
                        sims_for_this[out_key] = sim_configs[key]

                for magnet_config in manget_conf.keys():
                    output_filename = f"/eos/user/e/ekatrali/ecloud_plots/heat_load/{magnet_config}/{surface_curve}/{photoemission_stat}"
                    title = f"Heat load for {magnet_config} for Intensity: {intensity} [{surface_curve},{photoemission_stat}]"
                    os.makedirs(output_filename,exist_ok = True)
                    paramscan.plot_heat_load("SEY", sims_for_this, common_params={"Intensity":intensity, "Magnet Configuration": manget_conf[magnet_config]["Magnet Configuration"]}, 
                                             title=title, cmap = plt.cm.magma, curve_colors=sim_colors, top_lim = heat_load_lims[magnet_config],
                                             global_fontsize=13, show = show, savefig=save, output_filename=output_filename+f"/heatload_intens{intensity:.2f}_{magnet_config}.png",
                                             usetex=tex_available)
                    plt.close('all')
# paramscan.plot_half_cell_heat_load(magnets_in_half_cell, "SEY", "Intensity", common_params=sim_configs["Regular Photoemission"])
if generate_half_cell_load_plots:
    for intensity in paramscan.get_param_values("Intensity"):
        for surface_curve in ["CuO", "Cu2O"]:
            for photoemission_stat in ["Conditioned", "Unconditioned"]:
                sims_for_this = {}
                for key in sim_configs.keys():
                    if surface_curve in key and photoemission_stat in key:
                        out_key = key.replace(surface_curve,'')
                        out_key = out_key.replace(' ','')
                        out_key = out_key.replace('[','')
                        out_key = out_key.replace(']','')
                        out_key = out_key.replace(photoemission_stat,'')
                        sims_for_this[out_key] = sim_configs[key]

                title = f"Half cell Heat Load for Intensity: {intensity} [{surface_curve},{photoemission_stat}]"
                output_filename = f"/eos/user/e/ekatrali/ecloud_plots/heat_load/half_cell/{surface_curve}/{photoemission_stat}"
                os.makedirs(output_filename,exist_ok = True)
                paramscan.plot_half_cell_heat_load(magnets_in_half_cell, "SEY", sims_for_this, common_params={"Intensity": intensity}, curve_colors=sim_colors, unit = "W"
                                                   title=title, global_fontsize=13, show = show, savefig=save, output_filename=output_filename+f"/halfcell_intens{intensity:.2f}.png",
                                                   usetex=tex_available, ylabel = "Heat load [W/4920 bunches]")
                plt.close('all')
        
if plot_buildup:
    for magnet_config in manget_conf.keys():
        for intensity in paramscan.get_param_values("Intensity"):
            for sey in [None]: #paramscan.get_param_values("SEY"):
                for surface_curve in ["CuO", "Cu2O"]:
                    for photoemission_stat in ["Conditioned", "Unconditioned"]:
                        sims_for_this = {}
                        for key in sim_configs.keys():
                            if surface_curve in key and photoemission_stat in key:
                                out_key = key.replace(surface_curve,'')
                                out_key = out_key.replace(' ','')
                                out_key = out_key.replace('[','')
                                out_key = out_key.replace(']','')
                                out_key = out_key.replace(photoemission_stat,'')
                                sims_for_this[out_key] = sim_configs[key]
                            for sim in sims_for_this.keys():
                                title = f"Buildup as a function of SEY [Intensity:{intensity},{sim},{surface_curve},{photoemission_stat}]"
                                output_filename = f"/eos/user/e/ekatrali/ecloud_plots/buildup/{magnet_config}/{surface_curve}/{photoemission_stat}/{sim}/"
                                os.makedirs(output_filename,exist_ok = True)
                                paramscan.plot_buildup("SEY",common_params=sims_for_this[sim]|{"Intensity":intensity,"Magnet Configuration": manget_conf[magnet_config]["Magnet Configuration"]}, cmap = plt.cm.tab20c,
                                                                        bottom_lim= 10**8, title=title, global_fontsize=13, show_datapoints = False, returnfig = True, output_filename=output_filename+f"buildup_intens{intensity:.2f}.png",
                                                                        usetex=tex_available)
                                plt.semilogy()
                                plt.tight_layout()
                                if show:
                                    plt.show()
                                if save:
                                    plt.savefig(output_filename+f"buildup_intens{intensity:.2f}.png", dpi = 300)
                                plt.close('all')
