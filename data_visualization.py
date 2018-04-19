import stock
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import style

class DataVisualization:

    def __init__(self):
        self.Data = Stock(1,1,1)

        style.use('fivethirtyeight')
        self.fig = plt.figure()
        ax1 = self.fig.add_subplot(1,1,1)

    def show_prediction():    
        # initialization function: plot the background of each frame

# animation function.  This is called sequentially
def animate(i):

    x = np.linspace(0, 2, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
           frames=200, interval=20, blit=True)

plt.show()