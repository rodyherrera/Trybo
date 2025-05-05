from core.base_parser import BaseParser
import matplotlib.pyplot as plt
import numpy as np

class EnergyVisualizer:
    def __init__(self, parser: BaseParser):
        self.parser = parser
        
        timesteps, all_data, headers = parser.parse()

        self.timesteps = timesteps
        self.all_data = all_data
        self.headers = headers

        self.x_idx = headers.index('x')
        self.y_idx = headers.index('y')
        self.z_idx = headers.index('z')
        self.ke_idx = headers.index('c_ke_mobile')
        self.pe_idx = headers.index('c_pe_mobile')
        self.total_idx = headers.index('v_total_energy')

        self.average_ke = [np.mean(data[:, self.ke_idx]) for data in all_data]
        self.max_ke = [np.max(data[:, self.ke_idx]) for data in all_data]
        self.min_ke = [np.min(data[:, self.ke_idx]) for data in all_data]

        self.average_pe = [np.mean(data[:, self.pe_idx]) for data in all_data]
        self.max_pe = [np.max(data[:, self.pe_idx]) for data in all_data]
        self.min_pe = [np.min(data[:, self.pe_idx]) for data in all_data]

        self.average_total = [np.mean(data[:, self.total_idx]) for data in all_data]
        self.max_total = [np.max(data[:, self.total_idx]) for data in all_data]
        self.min_total = [np.min(data[:, self.total_idx]) for data in all_data]

    def plot_energy_evolution_statistics(self):
        fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

        # Kinetic Energy
        axs[0].plot(self.timesteps, self.average_ke, 'b--', label='Average')
        axs[0].fill_between(self.timesteps, self.min_ke, self.max_ke, color='blue', alpha=0.2, label='Min-Max Range')
        axs[0].set_ylabel('Kinetic Energy (eV)')
        axs[0].set_title('Evolution of Kinetic Energy')
        axs[0].grid(True, linestyle='--', alpha=0.7)
        axs[0].legend()

        # Potential Energy
        axs[1].plot(self.timesteps, self.average_pe, 'r--', label='Average')
        axs[1].fill_between(self.timesteps, self.min_pe, self.max_pe, color='red', alpha=0.2, label='Min-Max Range')
        axs[1].set_ylabel('Potential Energy (eV)')
        axs[1].set_title('Evolution of Potential Energy')
        axs[1].grid(True, linestyle='--', alpha=0.7)
        axs[1].legend()

        # Total Energy
        axs[2].plot(self.timesteps, self.average_total, 'g--', label='Average')
        axs[2].fill_between(self.timesteps, self.min_total, self.max_total, color='green', alpha=0.2, label='Min-Max Range')
        axs[2].set_xlabel('Timestep')
        axs[2].set_ylabel('Total Energy (eV)')
        axs[2].set_title('Evolution of the Total Energy')
        axs[2].grid(True, linestyle='--', alpha=0.7)
        axs[2].legend()

        plt.tight_layout()
        plt.savefig('energy_evolution.png', dpi=300)
        plt.show()
