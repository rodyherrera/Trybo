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

        self.statistics = self.calculate_statistics()

    def calculate_statistics(self):
        ke_stats = []
        pe_stats = []
        total_stats = []

        ke_idx = self.headers.index('c_ke_mobile')
        pe_idx = self.headers.index('c_pe_mobile')
        total_idx = self.headers.index('v_total_energy')
        
        for timestep_data in self.all_data:
            ke_values = timestep_data[:, ke_idx]
            pe_values = timestep_data[:, pe_idx]
            total_values = timestep_data[:, total_idx]

            ke_stats.append({
                'mean': np.mean(ke_values),
                'min': np.min(ke_values),
                'max': np.max(ke_values)
            })

            pe_stats.append({
                'mean': np.mean(pe_values),
                'min': np.min(pe_values),
                'max': np.max(pe_values)
            })
            
            total_stats.append({
                'mean': np.mean(total_values),
                'min': np.min(total_values),
                'max': np.max(total_values)
            })

        statistics = {
            'kinetic_energy': {
                'average': [stats['mean'] for stats in ke_stats],
                'min': [stats['min'] for stats in ke_stats],
                'max': [stats['max'] for stats in ke_stats]
            },
            'potential_energy': {
                'average': [stats['mean'] for stats in pe_stats],
                'min': [stats['min'] for stats in pe_stats],
                'max': [stats['max'] for stats in pe_stats]
            },
            'average': [stats['mean'] for stats in total_stats],
            'min': [stats['min'] for stats in total_stats],
            'max': [stats['max'] for stats in total_stats]
        }

        return statistics

    def plot_kinetic_energy(self):
        plt.figure(figsize=(12, 6))
        
        plt.plot(self.timesteps, self.statistics['kinetic_energy']['average'], 'b--', label='Average')
        plt.fill_between(
            self.timesteps, 
            self.statistics['kinetic_energy']['min'], 
            self.statistics['kinetic_energy']['max'],
            color='blue', 
            alpha=0.2, 
            label='Min-Max Range')
        plt.ylabel('Kinetic Energy (eV)')
        plt.title('Evolution of Kinetic Energy')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()

        plt.savefig('kinetic_energy_evolution.png', dpi=300)
        plt.show()

    def plot_potential_energy(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.timesteps, self.statistics['potential_energy']['average'], 'r--', label='Average')
        plt.fill_between(
            self.timesteps, 
            self.statistics['potential_energy']['min'],
            self.statistics['potential_energy']['max'],
            color='red',
            alpha=0.2,
            label='Min-Max Range')
        plt.xlabel('Timestep')
        plt.ylabel('Potential Energy (eV)')
        plt.title('Evolution of Potential Energy')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig('potential_energy_evolution.png', dpi=300)
        plt.show()

    def plot_total_energy(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.timesteps, self.statistics['average'], 'g--', label='Average')
        plt.fill_between(
            self.timesteps, 
            self.statistics['min'], 
            self.statistics['max'],
            color='green',
            alpha=0.2,
            label='Min-Max Range')
        plt.xlabel('Timestep')
        plt.ylabel('Total Energy (eV)')
        plt.title('Evolution of Total Energy')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()        

        plt.savefig('total_energy_evolution.png', dpi=300)
        plt.show()