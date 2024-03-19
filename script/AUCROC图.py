import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimSun']  #宋体
plt.rcParams['axes.unicode_minus'] = False
# Adjust the plot as per the new requirements

# Define the x values
x = np.linspace(0, 1, 400)

# Define the y values for the modified curve
y = np.sqrt(1 - (x - 1)**2)

# Recalculate the y values for y = x (linear)
y_linear = x

# Create the plot with specified color and line style changes
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='y = np.sqrt(1 - (x - 1)**2)', color='black', linewidth=2)  # Deep black color for the curve
plt.fill_between(x, 0, y, color='gray', alpha=0.3)  # Add gray shadow with some transparency
plt.plot(x, y_linear, color='gray', linewidth=2, linestyle=(0, (5, 10)))  # Gray dashed line with longer dashes

plt.text(0.6, 0.2, 'AUC', fontsize=20, ha='center', va='center')  # AUC text inside the gray area
plt.text(0.3, 0.8, 'ROC曲线', fontsize=20, ha='center', va='center')  # ROC Curve text above the curve

# Set the x and y axis limits to [0, 1]
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tick_params(axis='x', labelsize=15)  # Increase font size of x-axis tick labels
plt.tick_params(axis='y', labelsize=15)  # Increase font size of y-axis tick labels as well

plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('假正例率', fontsize=20)
plt.ylabel('真正例率', fontsize=20)
plt.tight_layout()
plt.show()
