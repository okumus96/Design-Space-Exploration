import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42   # TrueType – no Type 3 fonts
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import os



class Visualization:
    def __init__(self, save_dir=None):
        self.save_dir = save_dir
        if self.save_dir and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"Created directory for results: {self.save_dir}")

    def save_plot(self, filename, dpi=300, pad_inches=0.02):
        if self.save_dir:
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight', pad_inches=pad_inches)
            print(f"Saved plot to: {filepath}")
            plt.close()
        else:
            plt.show()

    def visualize_growth_curve(self, solutions, filename="cost_vs_growth.pdf"):
        """Minimum cost against the software growth level alpha, annotated with the number of control units."""
        pts = [(s['solve_info'].get('growth_alpha'), s['total_cost'], s['num_locations_used'], s['num_partitions'])
               for s in solutions if s.get('solve_info', {}).get('growth_alpha') is not None]
        if not pts:
            print("No growth-sweep solutions to visualize")
            return
        pts.sort()
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], '-o', color='#2a3f6f', lw=1.8, ms=6)
        for a, c, n, k in pts:
            ax.annotate(f"{n} units\n{k} partitions", (a, c), textcoords='offset points', xytext=(0, 9), ha='center', fontsize=8)
        ax.set_xlabel('software growth level $\\alpha$  (demand = today $\\times$ (1 + $\\alpha$ $\\cdot$ g[domain]))')
        ax.set_ylabel('minimum cost')
        ax.grid(alpha=0.3)
        ax.margins(y=0.2)
        fig.tight_layout()
        self.save_plot(filename)
        with open(os.path.join(self.save_dir, filename.replace('.pdf', '.dat')), 'w') as f:
            f.write("alpha cost units partitions\n")
            for a, c, n, k in pts:
                f.write(f"{a} {c:.2f} {n} {k}\n")
        plt.close(fig)

    def visualize_pareto_front(self, pareto_solutions, filename="pareto_front_analysis.pdf"):
        """
        Create a visualization of the Pareto front showing trade-offs.
        
        Args:
            pareto_solutions: List of solutions from optimize_pareto_epsilon_constraint
        """
        if not pareto_solutions:
            print("No solutions to visualize")
            return
        
        # Extract data
        cable_lengths = [sol['cable_length'] for sol in pareto_solutions]
        total_costs = [sol['total_cost'] for sol in pareto_solutions]
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 1, figsize=(14, 5))
        
        # Plot 1: Pareto Front (Cable Length vs Total Cost)
        ax1 = axes
        ax1.scatter(cable_lengths, total_costs, s=200, alpha=0.6, c='blue', edgecolors='black', linewidth=2)
        
        # Add solution numbers
        for i, (length, cost) in enumerate(zip(cable_lengths, total_costs), 1):
            ax1.annotate(f'S{i}', (length, cost), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=22, fontweight='bold')
        
        # Add line connecting solutions (Sorted by cable length)
        sorted_indices = sorted(range(len(cable_lengths)), key=lambda i: cable_lengths[i])
        sorted_lengths = [cable_lengths[i] for i in sorted_indices]
        sorted_costs = [total_costs[i] for i in sorted_indices]

        # Shade regions split by Pareto front (line remains untouched)
        x_span = (max(sorted_lengths) - min(sorted_lengths)) if len(sorted_lengths) > 1 else 1.0
        y_span = (max(sorted_costs) - min(sorted_costs)) if len(sorted_costs) > 1 else 1.0
        x_pad = 0.0
        y_pad = max(120.0, 0.04 * y_span)
        x_min = min(sorted_lengths) - x_pad
        x_max = max(sorted_lengths) + x_pad
        y_min = min(sorted_costs) - y_pad
        y_max = max(sorted_costs) + y_pad

        if len(sorted_lengths) >= 2:
            ax1.fill_between(
                sorted_lengths,
                sorted_costs,
                y_max,
                color="#000000",
                alpha=0.58
            )
            ax1.fill_between(
                sorted_lengths,
                y_min,
                sorted_costs,
                color="#000000",
                alpha=0.50
            )

        ax1.plot(sorted_lengths, sorted_costs, color='#1F2937', linestyle='--', alpha=0.95, linewidth=2.6)

        # Export Pareto points for TikZ/pgfplots usage
        base_name, _ = os.path.splitext(filename)
        points_filename = f"{base_name}_points.dat"
        if self.save_dir:
            points_path = os.path.join(self.save_dir, points_filename)
        else:
            points_path = points_filename

        with open(points_path, "w", encoding="utf-8") as points_file:
            points_file.write("# idx cable_length total_cost\n")
            for rank, (length, cost) in enumerate(zip(sorted_lengths, sorted_costs), 1):
                points_file.write(f"{rank} {float(length):.6f} {float(cost):.6f}\n")
        print(f"Saved Pareto points to: {points_path}")
        
        # Removed axis labels and title for publication-ready plot
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.margins(x=0.0, y=0.0)

        # Direct region labels (instead of legend)
        x_mid = 0.5 * (x_min + x_max)
        y_mid = 0.5 * (y_min + y_max)
        ax1.text(
            x_mid,
            y_mid - 0.22 * (y_max - y_min),
            'Dominated Region',
            fontsize=20,
            fontweight='bold',
            color="#008622",
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.75, edgecolor='none')
        )
        ax1.text(
            x_mid - 5.5,
            y_mid - 0.24 * (y_max - y_min),
            'Infeasible',
            fontsize=20,
            fontweight='bold',
            color="#7A7A7A",
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.75, edgecolor='none')
        )
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=0.05)
        self.save_plot(filename, dpi=600, pad_inches=0.0)
