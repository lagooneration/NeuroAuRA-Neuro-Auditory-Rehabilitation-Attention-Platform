import matplotlib.pyplot as plt
import numpy as np

# Set a beautiful, professional style
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif'

# Data
models = ['Subject-Specific\nLinear Baseline', 'Zero-Shot Global\nDeep Neural Network']
accuracies = [79.4, 73.1]
colors = ['#4A90E2', '#E94B3C']

fig, ax = plt.subplots(figsize=(8, 6))

# Create bars
bars = ax.bar(models, accuracies, color=colors, width=0.6, edgecolor='none', alpha=0.9)

# Add text labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, -25),  # 25 points vertical offset inside the bar
                textcoords="offset points",
                ha='center', va='bottom', color='white', fontweight='bold', fontsize=18)

# Formatting
ax.set_ylim(0, 100)
ax.set_ylabel('Decoding Accuracy (%)', fontweight='bold')
ax.set_title('Cross-Subject Zero-Shot Auditory Attention Decoding\n(n=16 Subjects, 2AFC)', fontweight='bold', pad=20)
ax.axhline(y=50, color='gray', linestyle='--', linewidth=2, label='Chance Level (50%)')

# Clean up axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig('poster_results_plot.png', dpi=300, transparent=False)
print("Plot saved to poster_results_plot.png")
