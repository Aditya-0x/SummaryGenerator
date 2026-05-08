import matplotlib.pyplot as plt
import numpy as np

# Set a clean, professional style for the presentation
plt.style.use('bmh')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ==========================================
# Chart 1: ROUGE Improvements
# ==========================================
metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
improvements = [0.0605, 0.0543, 0.0493]
colors = ['#2E5B88', '#4A90E2', '#89B4E8'] # Professional blue gradient

bars = ax1.bar(metrics, improvements, color=colors, width=0.5)
ax1.set_title('Performance Improvement over Base Model', fontsize=12, pad=15, fontweight='bold', color='#333333')
ax1.set_ylabel('Score Increase', fontsize=10, fontweight='bold')
ax1.set_ylim(0, 0.07)

# Add exact value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.0015,
             f'+{height:.4f}', ha='center', va='bottom', fontweight='bold', color='#333333')

# ==========================================
# Chart 2: Speed Trade-off
# ==========================================
models = ['Base Model', 'BART-large XSum']
times = [131, 1911]
colors2 = ['#9B9B9B', '#D0021B'] # Grey for base, Red for the heavy model

bars2 = ax2.bar(models, times, color=colors2, width=0.4)
ax2.set_title('Inference Time Trade-off', fontsize=12, pad=15, fontweight='bold', color='#333333')
ax2.set_ylabel('Time (Seconds)', fontsize=10, fontweight='bold')
ax2.set_ylim(0, 2300)

# Add exact time labels on top of bars
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 50,
             f'{int(height)}s', ha='center', va='bottom', fontweight='bold', color='#333333')

# Add a subtle background color to match the presentation template
fig.patch.set_facecolor('#F8F6F0')
ax1.set_facecolor('#F8F6F0')
ax2.set_facecolor('#F8F6F0')

# Save the high-res image
plt.tight_layout()
plt.savefig('c:/Users/adive/OneDrive/Desktop/summary/scratch/evaluation_chart.png', dpi=300, bbox_inches='tight')
print("Chart successfully saved!")
