import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import translations_Engl_Slo


# --------- Calculation ----------
def get_powers(Ug, Zg, Rb, Xb):
    Zb = complex(Rb, Xb)
    Rg = Zg.real
    Z_total = Zg + Zb
    Ib = Ug / Z_total
    Ub = Ib * Zb
    S_load = 0.5 * Ub * np.conj(Ib)
    P_load = S_load.real
    P_available = (np.abs(Ug) ** 2) / (8*Rg)
    P_reflected = P_available - P_load
    return Zb, S_load, P_available, P_load, P_reflected


# --------- GUI Class ----------
class PowerTransferApp:
    def __init__(self, root):
        self.root = root

        self.lang = tk.StringVar(value="sl")
        self.root.title(self.tr("Maximal Power Transfer Simulator"))

        # Layout with two frames
        self.left_frame = tk.Frame(root)
        self.left_frame.pack(side="left", fill="y", padx=10, pady=10)

        self.right_frame = tk.Frame(root)
        self.right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # --- Sliders with ticks ---
        self.Ugmag_var = tk.DoubleVar(value=5)
        self.Ugarg_var = tk.DoubleVar(value=0)
        self.Rg_var = tk.DoubleVar(value=10)
        self.Xg_var = tk.DoubleVar(value=10)
        self.Rb_var = tk.DoubleVar(value=15)
        self.Xb_var = tk.DoubleVar(value=10)

        self.create_slider(self.left_frame, "|Ug|", 0.0, 50.0, self.Ugmag_var, 0.1, 5.0)
        self.create_slider(self.left_frame, "Arg(Ug) [rad]", -math.pi, math.pi, self.Ugarg_var, 0.01, math.pi/2)
        self.create_slider(self.left_frame, "Rg [Ω]", 0, 40, self.Rg_var, step=0.1, tick_interval=10)
        self.create_slider(self.left_frame, "Xg [Ω]", -30, 30, self.Xg_var, step=0.1, tick_interval=10)
        self.create_slider(self.left_frame, "Rb [Ω]", 0, 40, self.Rb_var, step=0.1, tick_interval=10)
        self.create_slider(self.left_frame, "Xb [Ω]", -30, 30, self.Xb_var, step=0.11, tick_interval=10)

        # --- Output Text ---
        self.output_text = tk.Text(self.left_frame, height=8, width=40)
        self.output_text.pack(padx=5, pady=5)

        # --- Save button ---
        #self.save_button = ttk.Button(self.left_frame, text="Save Plot as PNG", command=self.save_plot)
        #self.save_button.pack(pady=5)

        # --- Axis control ---
        ttk.Label(self.left_frame, text=self.tr("X-axis (Real Ω):")).pack(anchor="w", padx=5)
        self.xmin = tk.DoubleVar(value=0)
        self.xmax = tk.DoubleVar(value=60)
        entry_frame_x = tk.Frame(self.left_frame)
        entry_frame_x.pack(padx=5, pady=2, fill="x")
        ttk.Entry(entry_frame_x, textvariable=self.xmin, width=7).pack(side="left")
        ttk.Entry(entry_frame_x, textvariable=self.xmax, width=7).pack(side="right")

        ttk.Label(self.left_frame, text=self.tr("Y-axis (Imag Ω):")).pack(anchor="w", padx=5)
        self.ymin = tk.DoubleVar(value=-30)
        self.ymax = tk.DoubleVar(value=30)
        entry_frame_y = tk.Frame(self.left_frame)
        entry_frame_y.pack(padx=5, pady=2, fill="x")
        ttk.Entry(entry_frame_y, textvariable=self.ymin, width=7).pack(side="left")
        ttk.Entry(entry_frame_y, textvariable=self.ymax, width=7).pack(side="right")


        self.update_axis_button = ttk.Button(self.left_frame, text=self.tr("Update Axis Limits"), command=self.update_plot)
        self.update_axis_button.pack(pady=5)

        # --- Matplotlib Figure ---
        self.fig, self.ax = plt.subplots() #figsize=(5, 3))
        self.ax.set_aspect('equal') # , adjustable='box')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.update_plot()

    # Language 
    def tr(self, key):
        return translations_Engl_Slo.translations[self.lang.get()].get(key, key)

    def create_slider(self, frame, label_text, from_, to, var, step, tick_interval):
        slider_frame = ttk.LabelFrame(frame, text=label_text, padding=5)
        slider_frame.pack(fill="x", pady=5)

        slider = tk.Scale(
            slider_frame,
            from_=from_,
            to=to,
            orient="horizontal",
            variable=var,
            resolution=step,
            tickinterval=tick_interval,
            length=260,
            command=lambda e: self.update_plot()
        )
        slider.pack(padx=10, pady=5)
        return slider

    def update_plot(self, event=None):
        Ugmag = self.Ugmag_var.get()
        Ugarg = self.Ugarg_var.get()
        Xg = self.Xg_var.get()
        Rg = self.Rg_var.get()
        Xg = self.Xg_var.get()
        Rb = self.Rb_var.get()
        Xb = self.Xb_var.get()


        Ug = Ugmag * np.exp(1j*Ugarg)
        Zg = Rg + 1j*Xg
        Zb, S_load, P_available, P_load, P_reflected = get_powers(Ug, Zg, Rb, Xb)
        #print (f"{Zb:.2f}", f"{S_load:.2f}", f"{P_load:.2f}", f"{P_reflected:.2f}")

        # Clear and redraw
        self.ax.clear()

        # Arrows from origin
        self.ax.arrow(0, 0, Zg.real, Zg.imag, head_width=0.8, head_length=1.2,
                      fc="red", ec="red", length_includes_head=True)
        self.ax.arrow(0, 0, Zb.real, Zb.imag, head_width=0.8, head_length=1.2,
                      fc="blue", ec="blue", length_includes_head=True)

        # Markers
        self.ax.plot(Zg.real, Zg.imag, 'ro', label=self.tr("Zg (Generator)"))
        self.ax.plot(Zb.real, Zb.imag, 'bo', label=self.tr("Zb (Load)"))

        # Axes
        self.ax.axhline(0, color='gray', linestyle='--')
        self.ax.axvline(0, color='gray', linestyle='--')
        self.ax.set_xlabel("Re")
        self.ax.set_ylabel("Im")
        self.ax.set_title(self.tr("Complex Plane: Zg and Zb"))
        self.ax.grid(True)
        self.ax.legend()

        # Axis limits from user
        try:
            self.ax.set_xlim(self.xmin.get(), self.xmax.get())
            self.ax.set_ylim(self.ymin.get(), self.ymax.get())
        except Exception:
            pass

        self.canvas.draw()

        # Update output text
        refl_pow_str = self.tr("Reflected Power")
        Gama = P_reflected / P_available
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END,
            f"Pg = {P_available:.3f}\n"
            f"Sb = {S_load:.3f} W\n"
            f"Pb = {P_load:.3f} W\n"
            f"P_ref= {P_reflected:.3f} W\n"   
            f"Γ ref= {Gama:.2f}"
        )
        #self.tr("Reflected Power") + f"= {P_reflected:.2f} W\n"

    #def save_plot(self):
    #    file_path = filedialog.asksaveasfilename(defaultextension=".png",
    #                                             filetypes=[("PNG Image", "*.png")])
    #    if file_path:
    #        self.fig.savefig(file_path)
    #        self.output_text.insert(tk.END, f"\nPlot saved to:\n{file_path}\n")

    
    def quit_app(self):
        try:
            plt.close(self.fig)
            self.root.quit()    # Stop the mainloop
            self.root.destroy() 
        except Exception:
            pass

# ---- Run the App ----
#if __name__ == "__main__":
#    root = tk.Tk()
#    app = PowerTransferApp(root)
#    root.mainloop()

