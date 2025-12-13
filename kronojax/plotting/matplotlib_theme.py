import pylab as plt
import matplotlib as mpl

my_theme = {
    'figure.facecolor': '#0C3C58',  # Background color of the figure (light yellow)
    'axes.facecolor': '#042420',  # Background color of the axes (light orange)
    'axes.edgecolor': '#ffffff',  # Color of the axes' borders (white)
    'axes.labelcolor': '#ffffff',  # Color of the axis labels (white)
    'text.color': '#ffffff',  # Color of the text (white)
    'xtick.color': '#ffffff',  # Color of the x-axis ticks (white)
    'ytick.color': '#ffffff',  # Color of the y-axis ticks (white)
    'lines.color': '#ffffff',  # Color of the lines in plots (white)
    'lines.linewidth': 1.0,  # Width of the lines in plots
    'lines.linestyle': '-',  # Style of the lines in plots (solid)
    'axes.grid': False,  # Display gridlines
    'legend.facecolor': '#0C584E',  # Background color of the legend (light orange)
    'legend.edgecolor': '#D89023',  # Color of the legend's border (dark orange)
    'legend.fontsize': 'medium',  # Font size of the legend
    'font.family': 'serif',  # Font family to use
    'font.size': 12.0,  # Default font size
    
}
mpl.style.library['alex'] = my_theme
plt.style.use('alex')
