![This is Trybo!](/screenshots/Trybo.png)

## Important Note on LAMMPS Compatibility
The standard LAMMPS installation from package managers (such as apt) or pre-compiled binaries may not include all the features required by Trybo. Specifically, Trybo requires LAMMPS to be compiled with GPU acceleration and several additional packages.

To ensure compatibility, we provide a build script (`build_lammps_gpu.sh`) that compiles LAMMPS with all necessary features. This script:

- Will not modify or replace any existing LAMMPS installation on your system
- Creates a custom LAMMPS executable in your current working directory
- Configures all required packages and optimizations for Trybo

You must use this custom executable when running Trybo simulations to avoid compatibility errors and to benefit from GPU acceleration.

#### 1. Wear Debris Analysis (Debris Clusters)
The `compute cluster/atom` command identifies connected groups of atoms based on a cutoff distance, assinging a unique cluster id to each atom. The results are saved periodically in `debris_clusters.dump`. This enables the detection and analysis of material fragments that may detach from the main bodies (nanoparticle or planes) during friction.

#### 2. Hotspot Analysis
This analysis specifically identifies and tracks atoms with particularly high kinetic energy ("hotspots") using `compute ke_hotspots` and a threshold (`variable hot_threshold`, `variable is_hotspot`). The script counts the total number of hotspots over time (`compute hotspots_count`, `fix hotspots_monitor`) and dumps per-atom kinetic energy and hotspot_status (`dump hotspots`).

#### 3. Energy Analysis
This analysis focuses on the distribution and evolution of kinetic, potential and total energy per atom for the "mobile" group of atoms (nanoparticle + flexible lower plane atoms). The `compute ke/atom` and `compute pe/atom` commands calcualte these per-atom energies, which are then summed to get total energy per atom (`variable total_energy`). This data is periodically saved via `dump energy_dump`.

#### 4. Material Transfer Analysis
Adhesives wear involves the transfer of material between contacting surfaces. The script uses dynamic groups and atom counts (`v_transferred_to_lower_group`, `v_transferred_to_upper_group`) to estimate the net transfer of atoms from the initial groups into regions defined as the lower and upper planes. This data is saved over time in `material_transfer.txt` and included in the thermo output.

#### 5. Structural Analysis (PTM)
Polyhedral Template Matching (PTM), calculated using `compute ptm/atom` and dumped per atom in `ptm.dump`, is an alternative method to CNA for identifying local crystal structures (FCC, HCP, BCC, surface, interface, unknown). It provides a detailed characterization of the local atomic envirionment and is particularly useful for analyzing structural changes during deformation and wear.

#### 6. Thermal Analysis (Nanoparticle & System)
Beyond the hotspot analysis, the script explicitly calculates the temperature of the `mobile_group` (`c_temp_mobile`) and specifically the temperature of the `nanoparticle_group` (`v_nanoparticle_temperature`) based on atomic velocities. These temperatures, along with the overal system temperature (`temp`), are available in the thermo output, and the nanoparticle temperature is saved in `wear_data.txt`. This allows for studying the thermal response of the system and the nanoparticle due to frictional heating and thermostatting.

#### 7. Stress Analysis (von Mises)
This analysis calcualtes the von Mises stress for each atom using `compute vonmises` and `variable atoms_stresss`. This scalar value is often used as an indicator of plastic deformation and yielding. The per-atom von Mises stress is dumped via `dump vonmises_dump`, and the average and maximum von Mises stress values are calculated over time (`compute vonmises_average`, `compute vonmises_max`) and saved in wear_data.txt.

#### 8. Friction and Contact Force Analysis
Focuses on the forces exchanged between the nanoparticle and the upper plane wich are fundamental to friction. The script calculates the normal force (`v_fz_nanoparticle_upper_plane`) and the tangential friction force (`v_fx_nanoparticle_upper_plane`) between the nanoparticle group and the upper plane group. It algo calculates the overall friction force on the upper plane (`v_fx_upper_plane`) and the coefficient of friction (`v_coefficient_friction`, `v_coefficient_friction_capped`). These values are available in thermo output and the coefficient of friction, is saved over time in `friction_data.txt`

#### 9. Wear Quantification (Mass Loss)
A direct measure of mechanical wear is the loss of material from the nanoparticle. The script tracks the number of atoms currently in the `nanoparticle_group` using `variable contact_count`. This count is saved over time in `contact_count.txt` and `wear_data.txt`, and is included in the thermo output.

#### 10. Crystal Structure Analysis (CNA - Common Neighbor Analaysis)
This analysis utilizes the `compute cna/atom` to assign a value to each atom indicating its local structural environment (e.g., perfect FCC, perfect HCP surface, or various defect types). The results are saved periodically for all atoms using `dump cna_dump`. The structural changes are key indicators of material degradation and deformation mechanisms that ocurr during wear processes.

#### 11. Atomic Coordination Analysis
The analysis is based on the `compute coord/atom` command, which calculates the number of nearest neighbors for each atom within a specified cutoff distance. The script dumps this per-atom coordination number (`c_coord`) via `dump coord_dump` and also calculates and saves the average, minimum and maximum coordination numbers over time using `compute coord_avg`, `compute coord_min`, `compute coord_max`, and `fix coord_stats`.

#### 12. Structural Defect Analysis (Centro-Symmetric)
The centro-symmetric parameter (`c_center_symmetric`), calculated using `compute centro/atom`, is a sensitive indicator of local structural deffects and disorder in crystalline material like FCC. The per-atom values are dumped in `center_symmetric.dump`, allowing for spatial analysis. The script also calculates and saves the average (`c_center_symmetric_average`), maximum (`c_center_symmetirc_max`), and a calculated defect percentage (`v_defect_percent`) based on this parameter over time in `wear_data.txt` and the thermo output.

#### 13. Radial Distribution Function (RDF) Analysis
The Radial Distribution Function (`c_radial_distribution_function[*]`), calculated using `compute rdf` and saved over time in `radial_distribution_function_output.txt`, describes the average atomic density as a function of distance from an atom. It provides information about the overall structural order and average interatomic distances in the system.

#### 14. System State and Dynamics Monitoring
The thermo output provides a snapshot of global system properites at regular intervals. This includes the simulation time step, total simulation time, total number of atoms, system pressure (`press`), total energy (`etotal`), potential energy (`pe`) and kinetic energy (`ke`). It also tracks the commanded x-position of the upper plane (`v_x_pos`) and its average z-position (`v_z_upper_plane`).