import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
import translations_Engl_Slo


def x_constant(t, X0=1.0, d=-1, A=1.0, w=1.0, phi=0.0): 
    return X0 * np.ones_like(t)

def x_impulse(t, X0=1.0, d=-1, A=1.0, w=1.0, phi=0.0):
    x = np.zeros_like(t)
    if len(t) > 1:
        x[0] = A / (t[1]-t[0])
    else:
        x[0] = A
    return x

def x_unitStep(t, X0=1.0, d=-1, A=1.0, w=1.0, phi=0.0): 
    return A * np.heaviside(t, 1.0)

def x_dumping(t, X0=1.0, d=-1, A=1.0, w=1.0, phi=0.0): 
    return X0 * np.exp(d * t)

def x_harmonic(t, X0=1.0, d=-1, A=1.0, w=1.0, phi=0.0): 
    return A * np.sin(w * t + phi)

def x_underdamped(t, X0=1.0, d=-1, A=1.0, w=1.0, phi=0.0): 
    return A * np.exp(d*t) * np.sin(w * t + phi)

def x_triangle(t, X0=1.0, d=-1, A=1.0, w=1.0, phi=0.0):
    t = np.asarray(t)
    x = np.zeros_like(t, dtype=float)
    T = 2*np.pi / w 

    mask1 = (t >= 0) & (t < T/4)
    x[mask1] = (4*X0/T) * t[mask1]

    mask2 = (t >= T/4) & (t < T/2)
    x[mask2] = 2*X0 - (4*X0/T)*t[mask2]
    return x

def x_square(t, X0=1.0, d=-1, A=1.0, w=1.0, phi=0.0):
    t = np.asarray(t)
    x = np.zeros_like(t, dtype=float)
    one = np.ones_like(t, dtype=float)
    T = 2*np.pi / w 

    mask1 = (t >= 0) & (t < T/2)
    x[mask1] = X0 * one[mask1]
    mask2 = (t >= T) & (t < 3*T/2)
    x[mask2] = X0 * one[mask2]
    mask3 = (t >= 2*T) & (t < 5*T/2)
    x[mask3] = X0 * one[mask3]
    mask4 = (t >= 3*T) & (t < 7*T/2)
    x[mask4] = X0 * one[mask4]
    return x


class FourierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fourier Series Interactive Demo")
        root.geometry("1100x700")

        self.lang = tk.StringVar(value="sl")

        self.input_signals = {
            self.tr("constant"): x_constant, 
            self.tr("impulse"): x_impulse,
            self.tr("unit-step"): x_unitStep,
            self.tr("dumping"): x_dumping,
            self.tr("harmonic"): x_harmonic,
            self.tr("underdamped"): x_underdamped,
            self.tr("triangle"): x_triangle,
            self.tr("square"): x_square
        }

        # Variables
        self.signal_choice = tk.StringVar(value=list(self.input_signals.keys())[7])
        self.period = tk.DoubleVar(value=2*np.pi)
        self.N_terms = tk.IntVar(value=5)
        self.X0_var = tk.DoubleVar(value=1.0)
        self.d_var = tk.DoubleVar(value=-1.0)
        self.A_var = tk.DoubleVar(value=1.0)
        self.w_var = tk.DoubleVar(value=1.0)
        self.phi_var = tk.DoubleVar(value=0.0)
        self.t_min = tk.DoubleVar(value=0)
        self.t_max = tk.DoubleVar(value=2*np.pi)

        # --- Layout ---
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel with canvas for scrolling
        left_container = ttk.Frame(main_frame)
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        self.canvas = tk.Canvas(left_container, width=250)
        self.canvas.pack(side=tk.LEFT, fill=tk.Y, expand=True)

        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=self.canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill="y")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.control_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0,0), window=self.control_frame, anchor="nw")

        # Bind to resize
        self.control_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # --- Controls ---
        ttk.Label(self.control_frame, text="Select Signal:").pack(anchor="w", pady=5)
        signal_menu = ttk.OptionMenu(
            self.control_frame,
            self.signal_choice,
            list(self.input_signals.keys())[0],
            *self.input_signals.keys(),
            command=lambda e: self.update_plots()
        )
        signal_menu.pack(anchor="w", fill=tk.X)

        ttk.Label(self.control_frame, text="Period T:").pack(anchor="w", pady=5)
        T_entry = ttk.Entry(self.control_frame, textvariable=self.period)
        T_entry.pack(anchor="w", fill=tk.X)

        # Horizontal sliders
        self.create_slider(self.control_frame, "Constant X0", 0.0, 5.0, self.X0_var, 0.1, 1.0)
        self.create_slider(self.control_frame, "Dumping δ", -10.0, 0, self.d_var, -1.0, 1.0)
        self.create_slider(self.control_frame, "Amplitude A", 0.0, 5.0, self.A_var, 0.1, 1.0)
        self.create_slider(self.control_frame, "Angular frequency ω", 0.0, 20.0, self.w_var, 0.1, 5.0)
        self.create_slider(self.control_frame, "Phase φ", -math.pi, math.pi, self.phi_var, 0.1, math.pi/2)
        self.create_slider(self.control_frame, "Number of terms N", 0, 50, self.N_terms, 1, 10)
        self.create_slider(self.control_frame, "t_min", -5, 0, self.t_min, 0.5, 1)
        self.create_slider(self.control_frame, "t_max", 0, 10, self.t_max, 0.5, 2)

        quit_btn = ttk.Button(self.control_frame, text="Quit", command=self.quit_app)
        quit_btn.pack(anchor="s", pady=10, fill=tk.X)

        # --- Right panel: plots ---
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig, (self.ax_time, self.ax_amp, self.ax_phase) = plt.subplots(3, 1, figsize=(7, 8))
        self.fig.subplots_adjust(hspace=0.6)

        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.update_plots()

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
            length=220,
            command=lambda v: self.update_plots()
        )
        slider.pack(padx=10, pady=5)
        return slider

    def signal(self, t):
        X0=self.X0_var.get(); d=self.d_var.get(); A=self.A_var.get()
        w=self.w_var.get(); phi=self.phi_var.get()
        choice = self.signal_choice.get()
        funcs = self.input_signals
        return funcs.get(choice, lambda t,*a: np.zeros_like(t))(t, X0, d, A, w, phi)

    def compute_fourier(self, t):
        T = self.period.get()
        w0 = 2*np.pi / T
        N = self.N_terms.get()
        x = self.signal(t)
        a0 = (2 / T) * np.trapezoid(x, t)
        a, b = [], []
        for n in range(1, N+1):
            cos_term = np.cos(n * w0 * t)
            sin_term = np.sin(n * w0 * t)
            a.append((2 / T) * np.trapezoid(x * cos_term, t))
            b.append((2 / T) * np.trapezoid(x * sin_term, t))
        y = a0/2 * np.ones_like(t)
        for n in range(1, N+1):
            y += a[n-1]*np.cos(n*w0*t) + b[n-1]*np.sin(n*w0*t)
        amps = [np.sqrt(a[i]**2 + b[i]**2) for i in range(N)]
        phases = [np.arctan2(b[i], a[i]) for i in range(N)]
        return x, y, amps, phases

    def update_plots(self):
        T = self.period.get()
        t_min, t_max = self.t_min.get(), self.t_max.get()
        if t_min >= t_max:
            t_max = t_min + T + 1e-3
        t_plt = np.linspace(t_min, t_max, 1000)
        t_fr = np.linspace(0, T, 1000)
        x, y, amps, phases = self.compute_fourier(t_fr)

        self.ax_time.clear(); self.ax_amp.clear(); self.ax_phase.clear()
        self.ax_time.plot(t_plt, self.signal(t_plt), label="Original Signal")
        self.ax_time.plot(t_plt, y, label=f"Reconstruction (N={self.N_terms.get()})")
        self.ax_time.set_title(f"Time Domain Signal ({t_min:.2f} to {t_max:.2f})")
        self.ax_time.set_xlabel("t"); self.ax_time.set_ylabel("Amplitude")
        self.ax_time.set_xlim(t_min, t_max); self.ax_time.legend(); self.ax_time.grid()

        indices = np.arange(1, len(amps)+1)
        if len(indices) > 0:
            self.ax_amp.stem(indices, amps, basefmt=" ", linefmt='C0-', markerfmt='C0o')
            self.ax_phase.stem(indices, phases, basefmt=" ", linefmt='C1-', markerfmt='C1s')
        self.ax_amp.set_xlabel("Harmonic Index (k)"); self.ax_amp.set_ylabel("Amplitude")
        self.ax_amp.set_title("Amplitude Spectrum"); self.ax_amp.grid()
        self.ax_phase.set_xlabel("Harmonic Index (k)"); self.ax_phase.set_ylabel("Phase (rad)")
        self.ax_phase.set_title("Phase Spectrum"); self.ax_phase.grid()

        self.canvas_fig.draw_idle()

    def quit_app(self):
        try: self.canvas_fig.get_tk_widget().destroy()
        except: pass
        try: self.fig.clear(); plt.close(self.fig)
        except: pass
        try: self.root.destroy()
        except: pass
