import einops
import numpy as np

import py_terminal_plotter as ptp

plotter = ptp.TerminalPlot(x_range=[0, 1], y_range=[0, 1])
plotter.create_axes()
plotter.setup_image(100, 100, z_range=[0, 10])

# Flip to r, c; and then flip to top left corner convention
embeddings = einops.rearrange(np.load("embeddings.npy"), 'x y n -> y x n')[::-1]

import sys, select, termios, time, tty
def getKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _settings)
    return key
_settings = termios.tcgetattr(sys.stdin)

px = 50
py = 50

def render(px, py):
    target = embeddings[py, px]
    distances = np.linalg.norm(embeddings - target, axis=2)
    distances[py, px] = 100
    plotter.plot_image_section(distances, start_row=0)
    plotter.draw()

render(px, py)
try:
    while True:
        key = getKey()
        if key == 'w':
            py += 1
            if py >= 100:
                py = 99
            render(px, py)
        if key == 's':
            py -= 1
            if py < 0:
                py = 0 
            render(px, py)
        if key == 'd':
            px += 1
            if px >= 100:
                px = 99 
            render(px, py)
        if key == 'a':
            px -= 1
            if px < 0:
                px = 0 
            render(px, py)
        time.sleep(0.05)

finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _settings)
