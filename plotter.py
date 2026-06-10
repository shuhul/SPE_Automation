import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import signal
from contextlib import contextmanager
import classifier


@contextmanager
def _sigint_default():
    """Temporarily restore the default SIGINT handler.

    matplotlib's Qt backend only sets up its SIGINT-wakeup socketpair (and
    the QSocketNotifier in _may_clear_sock) when a *custom* SIGINT handler is
    installed. automate.py installs one for Ctrl+C handling, which leaves
    that notifier alive watching a socket that gets closed once the window
    is closed — it then fires later during input(), raising
    OSError [WinError 10038] on Windows. Using the default handler during
    plt.show(block=True) makes matplotlib skip that setup entirely.
    """
    old_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old_handler)

def edges_from_centers(coords):
    diffs = np.diff(coords) / 2
    edges = np.concatenate(([coords[0] - diffs[0]], coords[:-1] + diffs, [coords[-1] + diffs[-1]]))
    return edges

def save_plot(foldername, scan_type, data_folder='data'):
    path = os.path.join(data_folder, foldername, scan_type)
    try:
        intensities = np.load(f'{path}/out.npy')
        wl = np.load(f'{path}/wl.npy')
        xs = np.load(f'{path}/xs.npy')
        ys = np.load(f'{path}/ys.npy')
        classified = np.load(f'{path}/classified.npy') if os.path.exists(f'{path}/classified.npy') else None
        summed = np.sum(intensities, axis=-1)
    except Exception as e:
        print(f"Error loading data for {foldername}: {e}")
        return

    if 'coarse' in scan_type or 'fine' in scan_type:

        x_edges = edges_from_centers(xs)
        y_edges = edges_from_centers(ys)

            
        fig_map, ax_map = plt.subplots(figsize=(6,6))
        im_map = ax_map.imshow(
            summed,
            extent=[x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]],
            origin='upper',
            cmap='viridis',
            aspect='equal'
        )
        ax_map.set_xlabel('X Position (um)')
        ax_map.set_ylabel('Y Position (um)')
        ax_map.set_title('PL Intensity Map')
        fig_map.colorbar(im_map, ax=ax_map, label='Peak Intensity (>550 nm)')
        
        if classified is not None:
            iys_true, ixs_true = np.where(classified == 1)
            ax_map.scatter(xs[ixs_true], ys[iys_true], facecolors='none', edgecolors='white', s=150, alpha=0.9, linewidths=2)

        # Save figure
        fig_map.savefig(f'{data_folder}/{foldername}/{scan_type}/pl_map.png', dpi=400)
        plt.close(fig_map)
        if classified is not None:
            save_dir = os.path.join(data_folder, foldername, scan_type, 'detection_spectrums')
            os.makedirs(save_dir, exist_ok=True)

            max_int = np.max(intensities)
            min_int = np.min(intensities)
            y_lim_bottom = min_int - (abs(min_int) * 0.1)
            y_lim_top = max_int * 1.1
            
            iys_true, ixs_true = np.where(classified == 1)
            
            for iy, ix in zip(iys_true, ixs_true):
                x_pos = xs[ix]
                y_pos = ys[iy]
                spec_data = intensities[iy, ix, :]
                
                # Create a dedicated figure for saving
                fig_temp, ax_temp = plt.subplots(figsize=(6, 6))
                
                ax_temp.plot(wl, spec_data, lw=2)
                ax_temp.set_xlabel('Wavelength')
                ax_temp.set_ylabel('Intensity')
                ax_temp.set_title(f'Spectrum at x={x_pos:.2f}, y={y_pos:.2f}')
                
                # Enforce the same global limits as the main GUI
                ax_temp.set_ylim(y_lim_bottom, y_lim_top)
                ax_temp.set_xlim(np.min(wl), np.max(wl))
                
                plt.tight_layout()
                
                filename = f"spec_x{x_pos:.2f}_y{y_pos:.2f}.png"
                fig_temp.savefig(os.path.join(save_dir, filename), dpi=400)
                plt.close(fig_temp) # Close immediately to free memory
                
            print("Done saving plots.")
    elif 'long' in scan_type:
        save_dir = os.path.join(data_folder, foldername, scan_type)
        os.makedirs(save_dir, exist_ok=True)

        max_int = np.max(intensities)
        min_int = np.min(intensities)
        y_lim_bottom = min_int - (abs(min_int) * 0.1)
        y_lim_top = max_int * 1.1
    
        spec_data = intensities[0, 0, :]
        x_pos = xs[0]
        y_pos = ys[0]
        
        # Create a dedicated figure for saving
        fig_temp, ax_temp = plt.subplots(figsize=(6, 6))
        
        ax_temp.plot(wl, spec_data, lw=2)
        ax_temp.set_xlabel('Wavelength')
        ax_temp.set_ylabel('Intensity')
        ax_temp.set_title(f'Spectrum at x={x_pos:.2f}, y={y_pos:.2f}')
        
        # Enforce the same global limits as the main GUI
        ax_temp.set_ylim(y_lim_bottom, y_lim_top)
        ax_temp.set_xlim(np.min(wl), np.max(wl))
        
        plt.tight_layout()
        
        filename = f"spec_x{x_pos:.2f}_y{y_pos:.2f}.png"
        fig_temp.savefig(os.path.join(save_dir, filename), dpi=600)
        plt.close(fig_temp) # Close immediately to free memory
        



def plot_heatmap(foldername, title='PL Spectrum', xlabel='X Position (um)', ylabel='Y Position (um)', 
                             heatmap_cmap='viridis', class1_color='red', data_folder='data', slideshow_interval=0.1):
    
    # --- 1. Load Data ---
    intensities = np.load(f'{data_folder}/{foldername}/out.npy')
    wl = np.load(f'{data_folder}/{foldername}/wl.npy')
    xs = np.load(f'{data_folder}/{foldername}/xs.npy')
    ys = np.load(f'{data_folder}/{foldername}/ys.npy')
    classified = np.load(f'{data_folder}/{foldername}/classified.npy')
    summed = np.sum(intensities, axis=-1)
    # plt.imsave(f'{data_folder}/{foldername}/pl_map.png', summed, cmap=heatmap_cmap)


    # --- 2. Setup Geometry ---
    def edges_from_centers(coords):
        diffs = np.diff(coords) / 2
        edges = np.concatenate(([coords[0] - diffs[0]], coords[:-1] + diffs, [coords[-1] + diffs[-1]]))
        return edges

    x_edges = edges_from_centers(xs)
    y_edges = edges_from_centers(ys)
    
    # Calculate pixel width/height for the cursor box
    dx = xs[1] - xs[0] if len(xs) > 1 else 1.0
    dy = ys[1] - ys[0] if len(ys) > 1 else 1.0
    
    # Grid dimensions
    n_rows = len(ys)
    n_cols = len(xs)

    # --- 3. Setup Plot ---
    fig, (ax_img, ax_spec) = plt.subplots(1, 2, figsize=(12, 6))
    
    im = ax_img.imshow(
        summed,
        extent=[x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]], 
        origin='upper',
        cmap=heatmap_cmap,
        aspect='equal'
    )
    ax_img.set_title(title)
    ax_img.set_xlabel(xlabel)
    ax_img.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax_img, label='Peak Intensity (>550 nm)')

    if classified is not None:
        iys, ixs = np.where(classified == 1)
        ax_img.scatter(xs[ixs], ys[iys], facecolors='none', edgecolors=class1_color,
                       s=100, linewidths=2, label='Class 1')
        ax_img.legend(loc='upper right')

    # Create the "Cursor" Box (Now Black)
    cursor_rect = patches.Rectangle((0,0), dx, dy, linewidth=4, edgecolor='black', facecolor='none', visible=False)
    ax_img.add_patch(cursor_rect)

    # Spectrum Plot Setup
    ax_spec.set_xlabel('Wavelength')
    ax_spec.set_ylabel('Intensity')
    
    # Pre-calculate limits
    max_intensity = np.max(intensities)
    min_intensity = np.min(intensities)
    ax_spec.set_ylim(min_intensity - (abs(min_intensity)*0.1), max_intensity * 1.1)
    ax_spec.set_xlim(np.min(wl), np.max(wl))

    (line,) = ax_spec.plot([], [], lw=2)
    
    # Status text
    status_text = ax_spec.text(0.5, 1.05, '', transform=ax_spec.transAxes, ha='center', va='bottom', 
                               color='black', fontweight='bold', fontsize=10)
    coord_text = ax_spec.text(0.5, 0.9, '', transform=ax_spec.transAxes, ha='center', 
                              bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # --- 4. State Management & Logic ---
    class Player:
        def __init__(self):
            self.ix = 0
            self.iy = 0
            self.is_playing = False
            self.timer = fig.canvas.new_timer(interval=int(slideshow_interval * 1000))  # Convert to milliseconds
            self.timer.add_callback(self.step_forward)
            self.update_view()

        def update_view(self):
            # Update Spectrum
            spectrum = intensities[self.iy, self.ix, :]
            line.set_data(wl, spectrum)
            
            # Update Cursor Box
            # x/y arrays are centers, so subtract half width to get bottom-left for Rectangle
            x_corner = xs[self.ix] - dx/2
            y_corner = ys[self.iy] - dy/2 
            
            cursor_rect.set_xy((x_corner, y_corner))
            cursor_rect.set_visible(True)
            
            # Update Texts
            if self.is_playing:
                status_str = "PLAYING (Press Space to Pause)"
            else:
                status_str = "PAUSED (Press Space to Play)"
            
            status_text.set_text(status_str)
            coord_text.set_text(f"x={xs[self.ix]:.2f}, y={ys[self.iy]:.2f}\n[Row {self.iy}, Col {self.ix}]")
            
            fig.canvas.draw_idle()

        def step_forward(self):
            # Linear sweep (Left/Right logic)
            self.ix += 1
            if self.ix >= n_cols:
                self.ix = 0
                self.iy += 1
                if self.iy >= n_rows:
                    self.iy = 0 
            self.update_view()

        def step_backward(self):
            # Linear sweep reverse
            self.ix -= 1
            if self.ix < 0:
                self.ix = n_cols - 1
                self.iy -= 1
                if self.iy < 0:
                    self.iy = n_rows - 1
            self.update_view()

        def move_up(self):
            # Spatial move up (decrease row index)
            self.iy -= 1
            if self.iy < 0:
                self.iy = n_rows - 1 # Wrap to bottom
            self.update_view()

        def move_down(self):
            # Spatial move down (increase row index)
            self.iy += 1
            if self.iy >= n_rows:
                self.iy = 0 # Wrap to top
            self.update_view()

        def toggle_play(self):
            if self.is_playing:
                self.timer.stop()
                self.is_playing = False
            else:
                self.timer.start()
                self.is_playing = True
            self.update_view()

        def set_pos(self, ix, iy):
            self.ix = ix
            self.iy = iy
            self.update_view()

    player = Player()

    # --- 5. Event Handlers ---

    def on_key(event):
        if event.key == 'right':
            player.step_forward()
        elif event.key == 'left':
            player.step_backward()
        elif event.key == 'up':
            player.move_up()
        elif event.key == 'down':
            player.move_down()
        elif event.key == ' ': # Spacebar
            player.toggle_play()

    def on_move(event):
        if event.inaxes != ax_img:
            return
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return

        ix_new = np.argmin(np.abs(xs - xdata))
        iy_new = np.argmin(np.abs(ys - ydata))

        if ix_new < 0 or ix_new >= n_cols or iy_new < 0 or iy_new >= n_rows:
            return
            
        if ix_new != player.ix or iy_new != player.iy:
            player.set_pos(ix_new, iy_new)

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.tight_layout()
    with _sigint_default():
        plt.show(block=True)







# def plot_heatmap_manual(foldername, scan_type,
#                         title='PL Spectrum',
#                           xlabel='X Position (um)', ylabel='Y Position (um)', 
#                              heatmap_cmap='viridis', class1_color='red', data_folder='data', slideshow_interval=0.1):
    
    
#     # --- Check if file exists to prevent immediate crash ---
#     file_path = os.path.join(data_folder, f'{foldername}/out.npy')
#     if not os.path.exists(file_path):
#         print(f"Error: Could not find {file_path}")
#         return

#     # --- 1. Load Data ---
#     intensities = np.load(f'{data_folder}/{foldername}/out.npy')
#     wl = np.load(f'{data_folder}/{foldername}/wl.npy')
#     xs = np.load(f'{data_folder}/{foldername}/xs.npy')
#     ys = np.load(f'{data_folder}/{foldername}/ys.npy')
#     classified = np.load(f'{data_folder}/{foldername}/classified.npy')
#     summed = np.sum(intensities, axis=-1)

#     if classified is not None:
#         save_dir = os.path.join(data_folder, foldername, 'detection_plots')
#         os.makedirs(save_dir, exist_ok=True)
        
#         # print(f"Generating classified spectrum plots in: {save_dir} ...")
        
#         # Pre-calculate limits so all plots share the same scale (matching your GUI logic)
#         max_int = np.max(intensities)
#         min_int = np.min(intensities)
#         y_lim_bottom = min_int - (abs(min_int) * 0.1)
#         y_lim_top = max_int * 1.1
        
#         iys_true, ixs_true = np.where(classified == 1)
        
#         # Turn off interactive mode so figures don't pop up during the loop
#         plt.ioff()
        
#         for iy, ix in zip(iys_true, ixs_true):
#             x_pos = xs[ix]
#             y_pos = ys[iy]
#             spec_data = intensities[iy, ix, :]
            
#             # Create a dedicated figure for saving
#             fig_temp, ax_temp = plt.subplots(figsize=(6, 6))
            
#             ax_temp.plot(wl, spec_data, lw=2)
#             ax_temp.set_xlabel('Wavelength')
#             ax_temp.set_ylabel('Intensity')
#             ax_temp.set_title(f'Spectrum at x={x_pos:.2f}, y={y_pos:.2f}')
            
#             # Enforce the same global limits as the main GUI
#             ax_temp.set_ylim(y_lim_bottom, y_lim_top)
#             ax_temp.set_xlim(np.min(wl), np.max(wl))
            
#             plt.tight_layout()
            
#             filename = f"spec_plot_x={x_pos:.2f}_y={y_pos:.2f}.png"
#             fig_temp.savefig(os.path.join(save_dir, filename), dpi=400)
#             plt.close(fig_temp) # Close immediately to free memory
            
#         print("Done saving plots.")
#         plt.ion() # Re-enable interactive mode if needed



#     # --- 2. Setup Geometry ---
#     def edges_from_centers(coords):
#         diffs = np.diff(coords) / 2
#         edges = np.concatenate(([coords[0] - diffs[0]], coords[:-1] + diffs, [coords[-1] + diffs[-1]]))
#         return edges

#     x_edges = edges_from_centers(xs)
#     y_edges = edges_from_centers(ys)

        
#     fig_map, ax_map = plt.subplots(figsize=(6,6))
#     im_map = ax_map.imshow(
#         summed,
#         extent=[x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]],
#         origin='upper',
#         cmap=heatmap_cmap,
#         aspect='equal'
#     )
#     ax_map.set_xlabel('X Position (um)')
#     ax_map.set_ylabel('Y Position (um)')
#     ax_map.set_title('PL Intensity Map')
#     fig_map.colorbar(im_map, ax=ax_map, label='Peak Intensity (>550 nm)')

#     # Save figure
#     fig_map.savefig(f'{data_folder}/{foldername}/pl_map.png', dpi=400)
#     plt.close(fig_map)
    
#     dx = xs[1] - xs[0] if len(xs) > 1 else 1.0
#     dy = ys[1] - ys[0] if len(ys) > 1 else 1.0
    
#     n_rows = len(ys)
#     n_cols = len(xs)

#    # --- 3. Setup Plot (Corrected) ---
#     fig, (ax_img, ax_spec) = plt.subplots(1, 2, figsize=(14, 7))
    
#     # Static Heatmap
#     im = ax_img.imshow(
#         summed,
#         extent=[x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]], 
#         origin='upper',
#         cmap=heatmap_cmap,
#         aspect='equal'
#     )
#     ax_img.set_title(title)
#     ax_img.set_xlabel(xlabel)
#     ax_img.set_ylabel(ylabel)
#     fig.colorbar(im, ax=ax_img, label='Peak Intensity (>550 nm)')

#     if classified is not None:
#         iys, ixs = np.where(classified == 1)
#         ax_img.scatter(xs[ixs], ys[iys], facecolors='none', edgecolors=class1_color,
#                        s=100, linewidths=2, label='Class 1')
#         ax_img.legend(loc='upper right')

#     # --- DYNAMIC ELEMENTS ---
#     # animated=True prevents them from being drawn in the standard loop
#     # We must manually draw them using ax.draw_artist()
#     cursor_rect = patches.Rectangle((0,0), dx, dy, linewidth=4, edgecolor='black', 
#                                     facecolor='none', visible=False, animated=True)
#     ax_img.add_patch(cursor_rect)

#     ax_spec.set_xlabel('Wavelength')
#     ax_spec.set_ylabel('Intensity')
    
#     max_intensity = np.max(intensities)
#     min_intensity = np.min(intensities)
#     # Ensure some padding so the line isn't on the edge
#     padding = (max_intensity - min_intensity) * 0.1 if max_intensity != min_intensity else 1.0
#     ax_spec.set_ylim(min_intensity - padding, max_intensity + padding)
#     ax_spec.set_xlim(np.min(wl), np.max(wl))

#     (line,) = ax_spec.plot(wl, intensities[0, 0, :], lw=2, animated=True)
    
#     status_text = ax_spec.text(0.5, 1.02, '', transform=ax_spec.transAxes, ha='center', va='bottom', 
#                                color='black', fontweight='bold', fontsize=12, animated=True)
#     coord_text = ax_spec.text(0.5, 0.95, '', transform=ax_spec.transAxes, ha='center', 
#                               bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'), animated=True)

#     # --- 4. Logic (Fixed Visibility) ---
#     class Player:
#         def __init__(self):
#             self.ix = 0
#             self.iy = 0
#             self.is_playing = False
#             self.bg = None 
            
#             self.timer = fig.canvas.new_timer(interval=int(slideshow_interval * 1000)) 
#             self.timer.add_callback(self.step_forward)

#             # Connect draw event to capture background
#             self.cid = fig.canvas.mpl_connect('draw_event', self.on_draw)

#         def on_draw(self, event):
#             """Called when the figure resizes or fully redraws."""
#             if event is not None and event.canvas != fig.canvas:
#                 return
            
#             # 1. Capture the clean background (heatmap + axes, no line/cursor)
#             self.bg = fig.canvas.copy_from_bbox(fig.bbox)
            
#             # 2. Manually draw the animated artists on top of the fresh background
#             # We do NOT use blit here because the standard draw cycle is still finishing.
#             self.draw_animated_artists()

#         def draw_animated_artists(self):
#             """Helper to update data and draw artists (no blitting logic here)."""
#             # Update Data
#             spectrum = intensities[self.iy, self.ix, :]
#             line.set_data(wl, spectrum)
            
#             x_corner = xs[self.ix] - dx/2
#             y_corner = ys[self.iy] - dy/2
#             cursor_rect.set_xy((x_corner, y_corner))
#             cursor_rect.set_visible(True)

#             if self.is_playing:
#                 status_str = "PLAYING (Press Space to Pause)"
#             else:
#                 status_str = "PAUSED (Press Space to Play)"
#             status_text.set_text(status_str)
#             coord_text.set_text(f"x={xs[self.ix]:.2f}, y={ys[self.iy]:.2f}")

#             # Draw Artists
#             ax_img.draw_artist(cursor_rect)
#             ax_spec.draw_artist(line)
#             ax_spec.draw_artist(status_text)
#             ax_spec.draw_artist(coord_text)

#         def update_view(self):
#             """Called during fast interaction (mouse move). Uses blitting."""
#             if self.bg is None:
#                 fig.canvas.draw_idle()
#                 return

#             # 1. Restore the background (wipes old cursor/line)
#             fig.canvas.restore_region(self.bg)
            
#             # 2. Draw the new positions
#             self.draw_animated_artists()
            
#             # 3. Blit to screen
#             fig.canvas.blit(fig.bbox)
#             fig.canvas.flush_events()

#         def step_forward(self):
#             self.ix += 1
#             if self.ix >= n_cols:
#                 self.ix = 0
#                 self.iy += 1
#                 if self.iy >= n_rows:
#                     self.iy = 0 
#             self.update_view()

#         def step_backward(self):
#             self.ix -= 1
#             if self.ix < 0:
#                 self.ix = n_cols - 1
#                 self.iy -= 1
#                 if self.iy < 0:
#                     self.iy = n_rows - 1
#             self.update_view()

#         def move_up(self):
#             self.iy -= 1
#             if self.iy < 0:
#                 self.iy = n_rows - 1
#             self.update_view()

#         def move_down(self):
#             self.iy += 1
#             if self.iy >= n_rows:
#                 self.iy = 0
#             self.update_view()

#         def toggle_play(self):
#             if self.is_playing:
#                 self.timer.stop()
#                 self.is_playing = False
#             else:
#                 self.timer.start()
#                 self.is_playing = True
#             self.update_view()

#         def set_pos(self, ix, iy):
#             if ix != self.ix or iy != self.iy:
#                 self.ix = ix
#                 self.iy = iy
#                 self.update_view()

#     player = Player()

#     # --- 5. Inputs (Optimized Calculation) ---
#     def on_key(event):
#         if event.key == 'right':
#             player.step_forward()
#         elif event.key == 'left':
#             player.step_backward()
#         elif event.key == 'up':
#             player.move_up()
#         elif event.key == 'down':
#             player.move_down()
#         elif event.key == ' ':
#             player.toggle_play()
#         elif event.key == 'q':
#             plt.close(fig) 

#     def on_move(event):
#         if event.inaxes != ax_img:
#             return
#         xdata, ydata = event.xdata, event.ydata
#         if xdata is None or ydata is None:
#             return

#         # Reverted to the robust method. 
#         # Since we fixed the graphics speed (blitting), this is fast enough 
#         # and guarantees the correct row is selected regardless of axis direction.
#         ix_new = np.argmin(np.abs(xs - xdata))
#         iy_new = np.argmin(np.abs(ys - ydata))

#         player.set_pos(ix_new, iy_new)

#     fig.canvas.mpl_connect("motion_notify_event", on_move)
#     fig.canvas.mpl_connect("key_press_event", on_key)

#     plt.tight_layout()
#     plt.show()
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_heatmap_manual(foldername, scan_type,
                        title='PL Spectrum',
                        xlabel='X Position (um)', ylabel='Y Position (um)', 
                        heatmap_cmap='viridis', class1_color='red', 
                        data_folder='data', slideshow_interval=0.1):
    
    # --- Check if file exists to prevent immediate crash ---
    base_path = os.path.join(data_folder, foldername, scan_type)
    file_path = os.path.join(base_path, 'out.npy')
    
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}")
        return

    # --- 1. Load Data ---
    intensities = np.load(os.path.join(base_path, 'out.npy'))
    wl = np.load(os.path.join(base_path, 'wl.npy'))
    xs = np.load(os.path.join(base_path, 'xs.npy'))
    ys = np.load(os.path.join(base_path, 'ys.npy'))
    summed = intensities.sum(axis=-1)

    # Safely load classified.npy in case the classifier hasn't run yet
    classified_path = os.path.join(base_path, 'classified.npy')
    classified = np.load(classified_path) if os.path.exists(classified_path) else None


    # --- 2. Generate Detection Plots (Autoscaled with Peak Info) ---
    if classified is not None:
        save_dir = os.path.join(base_path, 'detection_plots')
        os.makedirs(save_dir, exist_ok=True)
        
        iys_true, ixs_true = np.where(classified == 1)
        
        plt.ioff()
        
        for iy, ix in zip(iys_true, ixs_true):
            x_pos = xs[ix]
            y_pos = ys[iy]
            spec_data = intensities[iy, ix, :]
            
            fig_temp, ax_temp = plt.subplots(figsize=(6, 6))
            ax_temp.plot(wl, spec_data, lw=2)
            
            # Add Peak Lines and Text
            res = classifier.get_peak_annotation(spec_data, wl)
            if res:
                p_wl, p_int, l_wl, r_wl, fwhm = res
                ax_temp.axvline(p_wl, color='red', linestyle='--', alpha=0.7)
                ax_temp.plot([l_wl, r_wl], [p_int/2, p_int/2], color='orange', lw=2)
                ax_temp.text(0.95, 0.85, f"Center: {p_wl:.1f} nm\nFWHM: {fwhm:.1f} nm", 
                             transform=ax_temp.transAxes, ha='right', va='top', 
                             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
            
            ax_temp.set_xlabel('Wavelength')
            ax_temp.set_ylabel('Intensity')
            ax_temp.set_title(f'Spectrum at x={x_pos:.2f}, y={y_pos:.2f}')
            
            local_min = np.min(spec_data)
            local_max = np.max(spec_data)
            padding = (local_max - local_min) * 0.1 if local_max != local_min else 1.0
            
            ax_temp.set_ylim(local_min - padding, local_max + padding)
            ax_temp.set_xlim(np.min(wl), np.max(wl))
            
            plt.tight_layout()
            filename = f"spec_plot_x={x_pos:.2f}_y={y_pos:.2f}.png"
            fig_temp.savefig(os.path.join(save_dir, filename), dpi=400)
            plt.close(fig_temp)
            
        plt.ion()

    # --- 3. Setup Geometry ---
    def edges_from_centers(coords):
        if len(coords) < 2:
            return np.array([coords[0] - 0.5, coords[0] + 0.5])
        diffs = np.diff(coords) / 2
        edges = np.concatenate(([coords[0] - diffs[0]], coords[:-1] + diffs, [coords[-1] + diffs[-1]]))
        return edges

    x_edges = edges_from_centers(xs)
    y_edges = edges_from_centers(ys)

    # Map Figure
    fig_map, ax_map = plt.subplots(figsize=(6,6))
    im_map = ax_map.imshow(
        summed,
        extent=[x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]],
        origin='upper',
        cmap=heatmap_cmap,
        aspect='equal'
    )
    ax_map.set_xlabel('X Position (um)')
    ax_map.set_ylabel('Y Position (um)')
    ax_map.set_title('PL Intensity Map')
    fig_map.colorbar(im_map, ax=ax_map, label='Peak Intensity (>550 nm)')

    fig_map.savefig(os.path.join(base_path, 'pl_map.png'), dpi=400)
    plt.close(fig_map)
    
    dx = xs[1] - xs[0] if len(xs) > 1 else 1.0
    dy = ys[1] - ys[0] if len(ys) > 1 else 1.0
    
    n_rows = len(ys)
    n_cols = len(xs)

    # --- 4. Setup Interactive Plot ---
    fig, (ax_img, ax_spec) = plt.subplots(1, 2, figsize=(14, 7))
    
    im = ax_img.imshow(
        summed,
        extent=[x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]], 
        origin='upper',
        cmap=heatmap_cmap,
        aspect='equal'
    )
    ax_img.set_title(title)
    ax_img.set_xlabel(xlabel)
    ax_img.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax_img, label='Peak Intensity (>550 nm)')

    if classified is not None:
        iys, ixs = np.where(classified == 1)
        ax_img.scatter(xs[ixs], ys[iys], facecolors='none', edgecolors=class1_color,
                       s=100, linewidths=2, label='Class 1')
        ax_img.legend(loc='upper right')

    # FIX: Initialize cursor at the first data point, not (0,0)
    start_x = xs[0] - dx/2
    start_y = ys[0] - dy/2
    cursor_rect = patches.Rectangle((start_x, start_y), dx, dy, linewidth=4, edgecolor='black', 
                                    facecolor='none', visible=False)
    ax_img.add_patch(cursor_rect)

    ax_spec.set_xlabel('Wavelength')
    ax_spec.set_ylabel('Intensity')
    ax_spec.set_xlim(np.min(wl), np.max(wl))

    (line,) = ax_spec.plot(wl, intensities[0, 0, :], lw=2)
    
    # FIX: Initialize vertical line at a real wavelength, not 0
    peak_vline = ax_spec.axvline(wl[0], color='red', linestyle='--', alpha=0.7, visible=False)
    (fwhm_line,) = ax_spec.plot([], [], color='orange', lw=2, visible=False)
    
    status_text = ax_spec.text(0.5, 1.02, '', transform=ax_spec.transAxes, ha='center', va='bottom', 
                               color='black', fontweight='bold', fontsize=12)
    coord_text = ax_spec.text(0.5, 0.95, '', transform=ax_spec.transAxes, ha='center', 
                              bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
    peak_text = ax_spec.text(0.95, 0.85, '', transform=ax_spec.transAxes, ha='right', va='top', 
                             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'), visible=False)

    # Force strict axis limits just in case
    ax_img.set_xlim(np.min(x_edges), np.max(x_edges))
    ax_img.set_ylim(np.max(y_edges), np.min(y_edges))

    # --- 5. Logic ---
    class Player:
        def __init__(self):
            self.ix = 0
            self.iy = 0
            self.is_playing = False
            self.is_locked = False
            self.timer = fig.canvas.new_timer(interval=int(slideshow_interval * 1000)) 
            self.timer.add_callback(self.step_forward)

        def update_view(self):
            spectrum = intensities[self.iy, self.ix, :]
            line.set_data(wl, spectrum)
            
            # Handle Peak and FWHM Drawing
            is_class1 = (classified is not None) and (classified[self.iy, self.ix] == 1)
            if is_class1:
                res = classifier.get_peak_annotation(spectrum, wl)
                if res:
                    p_wl, p_int, l_wl, r_wl, fwhm = res
                    peak_vline.set_xdata([p_wl, p_wl])
                    fwhm_line.set_data([l_wl, r_wl], [p_int/2, p_int/2])
                    peak_text.set_text(f"Center: {p_wl:.1f} nm\nFWHM: {fwhm:.1f} nm")
                    
                    peak_vline.set_visible(True)
                    fwhm_line.set_visible(True)
                    peak_text.set_visible(True)
                else:
                    peak_vline.set_visible(False)
                    fwhm_line.set_visible(False)
                    peak_text.set_visible(False)
            else:
                peak_vline.set_visible(False)
                fwhm_line.set_visible(False)
                peak_text.set_visible(False)
            
            local_min = np.min(spectrum)
            local_max = np.max(spectrum)
            padding = (local_max - local_min) * 0.1 if local_max != local_min else 1.0
            ax_spec.set_ylim(local_min - padding, local_max + padding)

            x_corner = xs[self.ix] - dx/2
            y_corner = ys[self.iy] - dy/2
            cursor_rect.set_xy((x_corner, y_corner))
            
            if self.is_locked:
                cursor_rect.set_edgecolor('red')
                status_str = "LOCKED (Click to unlock)"
            else:
                cursor_rect.set_edgecolor('black')
                if self.is_playing:
                    status_str = "PLAYING (Press Space to Pause)"
                else:
                    status_str = "HOVERING (Click to lock)"
                
            cursor_rect.set_visible(True)
            status_text.set_text(status_str)
            coord_text.set_text(f"x={xs[self.ix]:.2f}, y={ys[self.iy]:.2f}")

            fig.canvas.draw_idle()

        def step_forward(self):
            self.ix += 1
            if self.ix >= n_cols:
                self.ix = 0
                self.iy += 1
                if self.iy >= n_rows:
                    self.iy = 0 
            self.update_view()

        def step_backward(self):
            self.ix -= 1
            if self.ix < 0:
                self.ix = n_cols - 1
                self.iy -= 1
                if self.iy < 0:
                    self.iy = n_rows - 1
            self.update_view()

        def move_up(self):
            self.iy -= 1
            if self.iy < 0:
                self.iy = n_rows - 1
            self.update_view()

        def move_down(self):
            self.iy += 1
            if self.iy >= n_rows:
                self.iy = 0
            self.update_view()

        def toggle_play(self):
            if self.is_playing:
                self.timer.stop()
                self.is_playing = False
            else:
                self.timer.start()
                self.is_playing = True
            self.update_view()

        def set_pos(self, ix, iy):
            if ix != self.ix or iy != self.iy:
                self.ix = ix
                self.iy = iy
                self.update_view()

    player = Player()

    # --- 6. Inputs ---
    def on_key(event):
        if event.key == 'right':
            player.step_forward()
        elif event.key == 'left':
            player.step_backward()
        elif event.key == 'up':
            player.move_up()
        elif event.key == 'down':
            player.move_down()
        elif event.key == ' ':
            player.toggle_play()
        elif event.key == 'q':
            plt.close(fig) 

    def on_move(event):
        if player.is_locked:
            return
            
        if event.inaxes != ax_img:
            return
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return

        ix_new = np.argmin(np.abs(xs - xdata))
        iy_new = np.argmin(np.abs(ys - ydata))
        player.set_pos(ix_new, iy_new)
        
    def on_click(event):
        if event.button == 1 and event.inaxes == ax_img:
            player.is_locked = not player.is_locked
            player.update_view()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.tight_layout()
    with _sigint_default():
        plt.show(block=True)


def select_emitters(foldername, scan_type,
                    title='PL Spectrum',
                    xlabel='X Position (um)', ylabel='Y Position (um)',
                    heatmap_cmap='viridis', data_folder='data', slideshow_interval=0.5):
    """Interactive heatmap for selecting which classified emitters to fine-scan.
    Identical to plot_heatmap_manual but right-click toggles emitter circles
    between Selected (red) and Ignored (gray). Close the window to confirm.
    Returns list of (x, y) tuples for selected emitters."""
    import matplotlib
    from matplotlib.lines import Line2D
    _switched_backend = False
    if matplotlib.get_backend().lower() == 'agg':
        try:
            plt.switch_backend('QtAgg')
        except Exception:
            plt.switch_backend('TkAgg')
        _switched_backend = True

    # --- Check if file exists to prevent immediate crash ---
    base_path = os.path.join(data_folder, foldername, scan_type)
    if not os.path.exists(os.path.join(base_path, 'out.npy')):
        print(f"Error: Could not find {base_path}/out.npy")
        return []

    # --- 1. Load Data ---
    intensities = np.load(os.path.join(base_path, 'out.npy'))
    wl  = np.load(os.path.join(base_path, 'wl.npy'))
    xs  = np.load(os.path.join(base_path, 'xs.npy'))
    ys  = np.load(os.path.join(base_path, 'ys.npy'))
    summed = intensities.sum(axis=-1)

    classified_path = os.path.join(base_path, 'classified.npy')
    classified = np.load(classified_path) if os.path.exists(classified_path) else None


    # --- 2. Generate Detection Plots ---
    if classified is not None:
        save_dir = os.path.join(base_path, 'detection_plots')
        os.makedirs(save_dir, exist_ok=True)
        iys_true, ixs_true = np.where(classified == 1)
        plt.ioff()
        for iy, ix in zip(iys_true, ixs_true):
            x_pos = xs[ix]
            y_pos = ys[iy]
            spec_data = intensities[iy, ix, :]
            fig_temp, ax_temp = plt.subplots(figsize=(6, 6))
            ax_temp.plot(wl, spec_data, lw=2)
            res = classifier.get_peak_annotation(spec_data, wl)
            if res:
                p_wl, p_int, l_wl, r_wl, fwhm = res
                ax_temp.axvline(p_wl, color='red', linestyle='--', alpha=0.7)
                ax_temp.plot([l_wl, r_wl], [p_int/2, p_int/2], color='orange', lw=2)
                ax_temp.text(0.95, 0.85, f"Center: {p_wl:.1f} nm\nFWHM: {fwhm:.1f} nm",
                             transform=ax_temp.transAxes, ha='right', va='top',
                             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
            ax_temp.set_xlabel('Wavelength')
            ax_temp.set_ylabel('Intensity')
            ax_temp.set_title(f'Spectrum at x={x_pos:.2f}, y={y_pos:.2f}')
            local_min = np.min(spec_data)
            local_max = np.max(spec_data)
            padding = (local_max - local_min) * 0.1 if local_max != local_min else 1.0
            ax_temp.set_ylim(local_min - padding, local_max + padding)
            ax_temp.set_xlim(np.min(wl), np.max(wl))
            plt.tight_layout()
            fig_temp.savefig(os.path.join(save_dir, f"spec_plot_x={x_pos:.2f}_y={y_pos:.2f}.png"), dpi=400)
            plt.close(fig_temp)
        plt.ion()

    # --- 3. Setup Geometry ---
    def edges_from_centers(coords):
        if len(coords) < 2:
            return np.array([coords[0] - 0.5, coords[0] + 0.5])
        diffs = np.diff(coords) / 2
        return np.concatenate(([coords[0] - diffs[0]], coords[:-1] + diffs, [coords[-1] + diffs[-1]]))

    x_edges = edges_from_centers(xs)
    y_edges = edges_from_centers(ys)

    # Map Figure
    fig_map, ax_map = plt.subplots(figsize=(6, 6))
    im_map = ax_map.imshow(summed,
                            extent=[x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]],
                            origin='upper', cmap=heatmap_cmap, aspect='equal')
    ax_map.set_xlabel('X Position (um)')
    ax_map.set_ylabel('Y Position (um)')
    ax_map.set_title('PL Intensity Map')
    fig_map.colorbar(im_map, ax=ax_map, label='Peak Intensity (>550 nm)')
    fig_map.savefig(os.path.join(base_path, 'pl_map.png'), dpi=400)
    plt.close(fig_map)

    dx = xs[1] - xs[0] if len(xs) > 1 else 1.0
    dy = ys[1] - ys[0] if len(ys) > 1 else 1.0
    n_rows = len(ys)
    n_cols = len(xs)

    # --- Emitter selection state ---
    if classified is not None:
        iys_e, ixs_e = np.where(classified == 1)
        emitter_xs = xs[ixs_e].astype(float)
        emitter_ys = ys[iys_e].astype(float)
    else:
        emitter_xs = np.array([])
        emitter_ys = np.array([])
    n_emitters = len(emitter_xs)
    selected = [True] * n_emitters

    # --- 4. Setup Interactive Plot ---
    fig, (ax_img, ax_spec) = plt.subplots(1, 2, figsize=(14, 7))

    im = ax_img.imshow(summed,
                       extent=[x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]],
                       origin='upper', cmap=heatmap_cmap, aspect='equal')
    ax_img.set_title(f'{title}  —  right-click emitter to toggle Selected / Ignored')
    ax_img.set_xlabel(xlabel)
    ax_img.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax_img, label='Peak Intensity (>550 nm)')

    scatter = None
    if n_emitters > 0:
        scatter = ax_img.scatter(emitter_xs, emitter_ys,
                                 facecolors='none', edgecolors=['red'] * n_emitters,
                                 s=100, linewidths=2, zorder=5)
        legend_elements = [
            Line2D([0], [0], marker='o', color='none', markerfacecolor='none',
                   markeredgecolor='red', markersize=10, markeredgewidth=2, label='Selected'),
            Line2D([0], [0], marker='o', color='none', markerfacecolor='none',
                   markeredgecolor='gray', markersize=10, markeredgewidth=2, label='Ignored'),
        ]
        ax_img.legend(handles=legend_elements, loc='upper right')

    def update_scatter():
        scatter.set_edgecolors(['red' if s else 'gray' for s in selected])
        fig.canvas.draw_idle()

    cursor_rect = patches.Rectangle((xs[0] - dx/2, ys[0] - dy/2), dx, dy,
                                     linewidth=4, edgecolor='black', facecolor='none', visible=False)
    ax_img.add_patch(cursor_rect)

    ax_spec.set_xlabel('Wavelength')
    ax_spec.set_ylabel('Intensity')
    ax_spec.set_xlim(np.min(wl), np.max(wl))

    (line,) = ax_spec.plot(wl, intensities[0, 0, :], lw=2)
    peak_vline = ax_spec.axvline(wl[0], color='red', linestyle='--', alpha=0.7, visible=False)
    (fwhm_line,) = ax_spec.plot([], [], color='orange', lw=2, visible=False)
    status_text = ax_spec.text(0.5, 1.02, '', transform=ax_spec.transAxes, ha='center', va='bottom',
                               color='black', fontweight='bold', fontsize=12)
    coord_text  = ax_spec.text(0.5, 0.95, '', transform=ax_spec.transAxes, ha='center',
                               bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
    peak_text   = ax_spec.text(0.95, 0.85, '', transform=ax_spec.transAxes, ha='right', va='top',
                               bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'), visible=False)

    ax_img.set_xlim(np.min(x_edges), np.max(x_edges))
    ax_img.set_ylim(np.max(y_edges), np.min(y_edges))

    # --- 5. Logic ---
    class Player:
        def __init__(self):
            self.ix = 0
            self.iy = 0
            self.is_playing = False
            self.is_locked = False
            self.timer = fig.canvas.new_timer(interval=int(slideshow_interval * 1000))
            self.timer.add_callback(self.step_forward)

        def update_view(self):
            spectrum = intensities[self.iy, self.ix, :]
            line.set_data(wl, spectrum)

            is_class1 = (classified is not None) and (classified[self.iy, self.ix] == 1)
            if is_class1:
                res = classifier.get_peak_annotation(spectrum, wl)
                if res:
                    p_wl, p_int, l_wl, r_wl, fwhm = res
                    peak_vline.set_xdata([p_wl, p_wl])
                    fwhm_line.set_data([l_wl, r_wl], [p_int/2, p_int/2])
                    peak_text.set_text(f"Center: {p_wl:.1f} nm\nFWHM: {fwhm:.1f} nm")
                    peak_vline.set_visible(True)
                    fwhm_line.set_visible(True)
                    peak_text.set_visible(True)
                else:
                    peak_vline.set_visible(False)
                    fwhm_line.set_visible(False)
                    peak_text.set_visible(False)
            else:
                peak_vline.set_visible(False)
                fwhm_line.set_visible(False)
                peak_text.set_visible(False)

            local_min = np.min(spectrum)
            local_max = np.max(spectrum)
            padding = (local_max - local_min) * 0.1 if local_max != local_min else 1.0
            ax_spec.set_ylim(local_min - padding, local_max + padding)

            cursor_rect.set_xy((xs[self.ix] - dx/2, ys[self.iy] - dy/2))
            if self.is_locked:
                cursor_rect.set_edgecolor('red')
                status_str = "LOCKED (Click to unlock)"
            else:
                cursor_rect.set_edgecolor('black')
                status_str = "PLAYING (Press Space to Pause)" if self.is_playing else "HOVERING (Click to lock)"
            cursor_rect.set_visible(True)
            status_text.set_text(status_str)
            coord_text.set_text(f"x={xs[self.ix]:.2f}, y={ys[self.iy]:.2f}")
            fig.canvas.draw_idle()

        def step_forward(self):
            self.ix += 1
            if self.ix >= n_cols:
                self.ix = 0
                self.iy += 1
                if self.iy >= n_rows:
                    self.iy = 0
            self.update_view()

        def step_backward(self):
            self.ix -= 1
            if self.ix < 0:
                self.ix = n_cols - 1
                self.iy -= 1
                if self.iy < 0:
                    self.iy = n_rows - 1
            self.update_view()

        def move_up(self):
            self.iy -= 1
            if self.iy < 0:
                self.iy = n_rows - 1
            self.update_view()

        def move_down(self):
            self.iy += 1
            if self.iy >= n_rows:
                self.iy = 0
            self.update_view()

        def toggle_play(self):
            if self.is_playing:
                self.timer.stop()
                self.is_playing = False
            else:
                self.timer.start()
                self.is_playing = True
            self.update_view()

        def set_pos(self, ix, iy):
            if ix != self.ix or iy != self.iy:
                self.ix = ix
                self.iy = iy
                self.update_view()

    player = Player()

    # --- 6. Inputs ---
    def on_key(event):
        if event.key == 'right':
            player.step_forward()
        elif event.key == 'left':
            player.step_backward()
        elif event.key == 'up':
            player.move_up()
        elif event.key == 'down':
            player.move_down()
        elif event.key == ' ':
            player.toggle_play()
        elif event.key == 'q':
            plt.close(fig)

    def on_move(event):
        if player.is_locked or event.inaxes != ax_img:
            return
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return
        player.set_pos(int(np.argmin(np.abs(xs - xdata))),
                       int(np.argmin(np.abs(ys - ydata))))

    def on_click(event):
        if event.inaxes != ax_img:
            return
        if event.button == 1:
            player.is_locked = not player.is_locked
            player.update_view()
        elif event.button == 3 and n_emitters > 0:
            if event.xdata is None or event.ydata is None:
                return
            dists = np.sqrt((emitter_xs - event.xdata)**2 + (emitter_ys - event.ydata)**2)
            nearest = int(np.argmin(dists))
            if dists[nearest] < max(dx, dy):
                selected[nearest] = not selected[nearest]
                update_scatter()

    fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.tight_layout()
    with _sigint_default():
        plt.show(block=True)

    if _switched_backend:
        plt.switch_backend('Agg')

    return [(float(emitter_xs[i]), float(emitter_ys[i])) for i in range(n_emitters) if selected[i]]


def open_heatmap(foldername, scan_type, data_folder='data', **kwargs):
    """Like plot_heatmap_manual but forces an interactive window even from scripts.
    Temporarily switches away from Agg if needed, and switches back to Agg
    once the window is closed (avoids a stale Qt socket notifier from
    plt.show(block=True) firing during a later input() call on Windows)."""
    import matplotlib
    backend = matplotlib.get_backend()
    switched = False
    if backend.lower() == 'agg':
        try:
            plt.switch_backend('QtAgg')
        except Exception:
            plt.switch_backend('TkAgg')
        switched = True
    try:
        plot_heatmap_manual(foldername, scan_type, data_folder=data_folder, **kwargs)
    finally:
        if switched:
            plt.switch_backend('Agg')
            # Drop the Qt interrupt-handling wakeup socket registered by
            # plt.show(block=True); otherwise its QSocketNotifier fires on
            # the now-closed socket later (e.g. during input()), raising
            # OSError [WinError 10038] in _may_clear_sock on Windows.
            try:
                import signal
                signal.set_wakeup_fd(-1)
            except (ValueError, OSError):
                pass