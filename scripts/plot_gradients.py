import einops
import numpy as np

import py_terminal_plotter as ptp

plotter = ptp.TerminalPlot(x_range=[0, 1], y_range=[0, 1])
plotter.create_axes(title="Embedding Laplacian")
plotter.setup_image(100, 100, z_range=[0, 10])

# Flip to r, c; and then flip to top left corner convention
embeddings = einops.rearrange(np.load("embeddings.npy"), 'x y n -> y x n')[::-1]

gradients = np.zeros(embeddings.shape[:2])
for i in range(1, 99):
    for j in range(1, 99):
        laplacian = embeddings[i+1, j] + embeddings[i-1, j] + embeddings[i, j+1] + embeddings[i, j+1] - 4*embeddings[i, j]
        gradients[i, j] = np.linalg.norm(laplacian)

#slice_plotter = ptp.TerminalPlot(x_range=[0, 1], y_range=[0, np.max(gradients)])
slice_plotter = ptp.TerminalPlot(x_range=[0, 1], y_range=[0, 10])
slice_plotter.create_axes(title="Embedding Laplacian")

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
    x = px / 99
    y = py / 99
    display = np.copy(gradients)
    display[py, px] = 100
    plotter.set_title(f"Embedding Laplacian ({x:.3f}, {y:.3f}) |L|={gradients[py, px]:.5f}")
    plotter.plot_image_section(display, start_row=0)
    plotter.draw()

def render_x_slice(py):
    xs = np.linspace(0, 1, 100)
    ys = gradients[py, :]
    y = py / 99
    slice_plotter.set_title(f"Embedding Laplacian X-slice (Y={y:.3f})")
    slice_plotter.clear_plot_area()
    slice_plotter.max_binned_plot(xs, ys)
    slice_plotter.draw()

def render_y_slice(px):
    xs = np.linspace(0, 1, 100)
    ys = gradients[:, px]
    x = px / 99
    slice_plotter.set_title(f"Embedding Laplacian Y-slice (X={x:.3f})")
    slice_plotter.clear_plot_area()
    slice_plotter.max_binned_plot(xs, ys)
    slice_plotter.draw()

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
        if key == 'x':
            render_x_slice(py)
        if key == 'y':
            render_y_slice(px)
        if key == 'q':
            break
        time.sleep(0.05)

finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _settings)
