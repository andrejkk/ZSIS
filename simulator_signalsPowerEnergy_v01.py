import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
from scipy.optimize import curve_fit
from signals_definitions import get_signals
from systemsTime_definitions import get_systems
import translations_Engl_Slo




def get_Awfi(t, x):

    """
    Estimate amplitude A, period T_est (in same units as t), and phase phi
    of a sampled cosine signal using FFT + curve fitting.
    """
    N = len(x)

    # Estimate mean sampling interval
    dt_mean = np.mean(np.diff(t))


     # model function
    def cos_model(t, A, w, phi):
        return A * np.cos(w*t + phi)

    # initial guesses
    A0 = (x.max() - x.min()) / 2
    # rough frequency guess using FFT

    freqs = np.fft.rfftfreq(len(t), d=dt_mean)
    fft_mags = np.abs(np.fft.rfft(x))
    freq_guess = freqs[np.argmax(fft_mags[1:])+1]  # avoid DC
    w0 = 2*np.pi*freq_guess
    phi0 = 0.0

    # fit
    popt, _ = curve_fit(cos_model, t, x, p0=[A0, w0, phi0])
    A_est, w_est, phi_est = popt
    T_est = 2*np.pi / w_est

    return A_est, T_est, phi_est

# A_est, T_est, phi_est =  get_Awfi(t, x)



# --- GUI class ---
class TimeDomainPowerEnergySimulator:
    def __init__(self, root):
        self.root = root
        root.geometry("1100x750")

        self.lang = tk.StringVar(value="sl")
        root.title(self.tr("Time-domain Power/Energy Simulator"))


        sel_signals = [self.tr("Constant"), self.tr("Impulse"), self.tr("Unit-step"), 
                       self.tr("Harmonic"), self.tr("Triangle"), self.tr("Square"), self.tr("DoubleTriangle")]
        all_input_signals = get_signals(self.tr)
        self.input_signals = {k: all_input_signals[k] for k in sel_signals if k in all_input_signals}
        
        self.systems = get_systems(self.tr)

        # Signal and system dict
        '''
        self.systems = {
            self.tr("Resistor, x=i, y=u"): sys_resistor_xi_yu_y,
            self.tr("Resistor, x=u, y=i"): sys_resistor_xu_yi_y,
            self.tr("Capacitor, x=i, y=u"): sys_capacitor_xi_yu_y,
            self.tr("Capacitor, x=u, y=i"): sys_capacitor_xi_yu_y,
            self.tr("Inductor, x=i, y=u"): sys_inductor_xi_yu_y,
            self.tr("Inductor, x=u, y=i"): sys_inductor_xi_yu_y,
            self.tr("Serial RC, x=i, y=u"): sys_serialRC_xi_yu_y
        }
        '''

        '''
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
        '''



        # Variables
        
        self.system_var = tk.StringVar(value=list(self.systems.keys())[0])
        self.input_sig_var = tk.StringVar(value=list(self.input_signals.keys())[0])
        self.R_var = tk.DoubleVar(value=1.0)
        self.L_var = tk.DoubleVar(value=1.0)
        self.C_var = tk.DoubleVar(value=1.0)

        self.C_u0_var = tk.DoubleVar(value=0.0)
        self.L_i0_var = tk.DoubleVar(value=0.0)

        self.X0_var = tk.DoubleVar(value=1.0)
        self.d_var = tk.DoubleVar(value=-1.0)
        self.A_var = tk.DoubleVar(value=1.0)
        self.w_var = tk.DoubleVar(value=1.0)
        self.phi_var = tk.DoubleVar(value=0.0)
        self.t1_var = tk.DoubleVar(value=0.0)
        self.t2_var = tk.DoubleVar(value=5.0)
        self.t1_choice = tk.StringVar(value="custom")
        self.t2_choice = tk.StringVar(value="custom")
        self.tmin_var = tk.DoubleVar(value=0.0)
        self.tmax_var = tk.DoubleVar(value=5.0)
        self.dt_var = tk.DoubleVar(value=0.01)



        # --- LEFT SCROLLABLE FRAME ---
        self.left_container = tk.Frame(root)
        self.left_container.pack(side="left", padx=10, pady=10, fill="y")

        self.left_canvas = tk.Canvas(self.left_container)
        self.left_scrollbar = ttk.Scrollbar(self.left_container, orient="vertical", command=self.left_canvas.yview)
        self.left_scrollable = tk.Frame(self.left_canvas)

        self.left_scrollable.bind(
            "<Configure>",
            lambda e: self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))
        )

        self.left_canvas.create_window((0, 0), window=self.left_scrollable, anchor="nw")
        self.left_canvas.configure(yscrollcommand=self.left_scrollbar.set)

        self.left_canvas.pack(side="left", fill="y", expand=True)
        self.left_scrollbar.pack(side="right", fill="y")

        self.left_frame = self.left_scrollable

        # --- CONTROLS ---
        f_input = ttk.LabelFrame(self.left_frame, text=self.tr("input_signal"), padding=6)
        f_input.pack(fill="x", pady=4)
        ttk.OptionMenu(f_input, self.input_sig_var, list(self.input_signals.keys())[0], *self.input_signals.keys()).pack(fill="x", padx=4, pady=4)

            
        #f_sys = ttk.LabelFrame(self.left_frame, text="Select system", padding=6)
        f_sys = ttk.LabelFrame(self.left_frame, text=self.tr("select_system"), padding=6)
        f_sys.pack(fill="x", pady=4)
        ttk.OptionMenu(f_sys, self.system_var, list(self.systems.keys())[0], *self.systems.keys()).pack(fill="x", padx=4, pady=4)
        
        #create_slider(self, frame, label_text, from_, to, var, step, tick_interval)
        f_input = ttk.LabelFrame(self.left_frame, text="Signal parameters", padding=6)
        self.create_slider(f_input, self.tr("Constant X0"), 0.0, 5.0, self.X0_var, 0.1, 1.0)
        self.create_slider(f_input, self.tr("Dumping δ"), -10.0, 0, self.d_var, -1.0, 1.0)
        self.create_slider(f_input, self.tr("Amplitude A"), 0.0, 5.0, self.A_var, 0.1, 1.0)
        self.create_slider(f_input, self.tr("Angular frequency ω"), 0.0, 20.0, self.w_var, 0.1, 5.0)
        self.create_slider(f_input, self.tr("Phase φ"), -math.pi, math.pi, self.phi_var, 0.01, math.pi/2)

        f_rlc = ttk.LabelFrame(self.left_frame, text="System parameters (R,L,C)", padding=6)
        f_rlc.pack(fill="x", pady=4)
        self.create_slider(f_rlc, "R [Ω]", 0.0, 10.0, self.R_var, 0.1, 2.0)
        self.create_slider(f_rlc, "L [H]", 0.0, 10.0, self.L_var, 0.1, 2.0)
        self.create_slider(f_rlc, self.tr("i_L(0) [A]"), -10.0, 10.0, self.L_i0_var, 0.1, 1.0)
        self.create_slider(f_rlc, "C [F]", 0.0, 10.0, self.C_var, 0.1, 2.0)
        self.create_slider(f_rlc, self.tr("u_C(0) [V]"), -10.0, 10.0, self.C_u0_var, 0.1, 1.0)

        f_tsel = ttk.LabelFrame(self.left_frame, text=self.tr("Integration interval [t1, t2]"), padding=6)
        f_tsel.pack(fill="x", pady=4)
        ttk.OptionMenu(f_tsel, self.t1_choice, self.tr("custom"), self.tr("custom"), "-∞").pack(fill="x", padx=4, pady=2)
        self.create_slider(f_tsel, "t1", -5.0, 20.0, self.t1_var, 0.05, 5.0)
        ttk.OptionMenu(f_tsel, self.t2_choice, self.tr("custom"), self.tr("custom"), "∞").pack(fill="x", padx=4, pady=2)
        self.create_slider(f_tsel, "t2", 0.0, 20.0, self.t2_var, 0.1, 5.0)

        f_time = ttk.LabelFrame(self.left_frame, text=self.tr("Time window"), padding=6)
        f_time.pack(fill="x", pady=4)
        self.create_slider(f_time, self.tr("Min t [s]"), -10, 10.0, self.tmin_var, 0.1, 5.0)
        self.create_slider(f_time, self.tr("Max t [s]"), 0.1, 20.0, self.tmax_var, 0.1, 5.0)
        self.create_slider(f_time, "dt [s]", 0.001, 0.1, self.dt_var, 0.001, 0.02)

        # --- RIGHT FRAME ---
        # --- RIGHT FRAME WITH SCROLLBAR ---
        right_container = tk.Frame(root)
        right_container.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.right_canvas = tk.Canvas(right_container)
        self.right_scrollbar = ttk.Scrollbar(right_container, orient="vertical",
                                             command=self.right_canvas.yview)
        self.right_scrollable = tk.Frame(self.right_canvas)

        self.right_scrollable.bind(
            "<Configure>",
            lambda e: self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all"))
        )

        self.right_canvas.create_window((0, 0), window=self.right_scrollable, anchor="nw")
        self.right_canvas.configure(yscrollcommand=self.right_scrollbar.set)

        self.right_canvas.pack(side="left", fill="both", expand=True)
        self.right_scrollbar.pack(side="right", fill="y")

        right = self.right_scrollable  # <- use this as parent for widgets below

        # --- PLOT ---
        self.fig = Figure(figsize=(7, 6), dpi=100)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)

        self.ax1.set_xlabel("t [s]")
        self.ax1.set_ylabel("x(t), y(t)")
        #self.ax1.set_ylim([-10, 10])
        self.ax1.grid(True)
        self.ax2.set_xlabel("t [s]")
        self.ax2.set_ylabel(self.tr("Power p(t)"))
        self.ax2.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # --- LABELS ---
        self.label_InputPW = ttk.Label(right, text=self.tr("Input signal P, W, S"), font=("Helvetica", 12))
        self.label_InputPW.pack(anchor="w")
        self.label_x_P = ttk.Label(right, text=self.tr("input P") + " = ", font=("Helvetica", 12))
        self.label_x_P.pack(anchor="w")
        self.label_x_W = ttk.Label(right, text=self.tr("input W") + " = ", font=("Helvetica", 12))
        self.label_x_W.pack(anchor="w")
        self.label_x_S = ttk.Label(right, text=self.tr("input S") + " = ", font=("Helvetica", 12))
        self.label_x_S.pack(anchor="w")

        self.label_ResponsePW = ttk.Label(right, text=self.tr("Response signal P, W, S"), font=("Helvetica", 12))
        self.label_ResponsePW.pack(anchor="w")
        self.label_y_P = ttk.Label(right, text=self.tr("response P") + " = ", font=("Helvetica", 12))
        self.label_y_P.pack(anchor="w")
        self.label_y_W = ttk.Label(right, text=self.tr("response W") + " = ", font=("Helvetica", 12))
        self.label_y_W.pack(anchor="w")
        self.label_y_S = ttk.Label(right, text=self.tr("response S") + " = ", font=("Helvetica", 12))
        self.label_y_S.pack(anchor="w")

        self.label_NetworkPW = ttk.Label(right, text=self.tr("One-port network P, W, S"), font=("Helvetica", 12))
        self.label_NetworkPW.pack(anchor="w")
        self.label_xy_P = ttk.Label(right, text=self.tr("network P") + " = ", font=("Helvetica", 12))
        self.label_xy_P.pack(anchor="w")
        self.label_xy_W = ttk.Label(right, text=self.tr("network W") + " = ", font=("Helvetica", 12))
        self.label_xy_W.pack(anchor="w")
        self.label_xy_S = ttk.Label(right, text=self.tr("network S") + " = ", font=("Helvetica", 12))
        self.label_xy_S.pack(anchor="w")

        # Trace live updates
        for var in [self.system_var, self.input_sig_var,
                    self.R_var, self.L_var, self.L_i0_var, self.C_var, self.C_u0_var,
                    self.X0_var, self.d_var, self.A_var, self.w_var, self.phi_var,
                    self.t1_var, self.t2_var,
                    self.t1_choice, self.t2_choice,
                    self.tmin_var, self.tmax_var, self.dt_var]:
            var.trace_add("write", lambda *a: self.update_plot())

        self.update_plot()

    # Language 
    def tr(self, key):
        return translations_Engl_Slo.translations[self.lang.get()].get(key, key)

    
    # --- Input signals ---
    def make_input_signal(self, kind, t, X0=1.0, d=-1, A=1.0, w=1.0, phi=0.0):

        x = self.input_signals[kind](t, X0, d, A, w, phi)
        return x

        '''
        if kind == self.tr("constant"):
            #signal_func = self.input_signals["harmonic"]
            x = self.input_signals[self.tr("constant")](t, X0, d, A, w, phi)
            return x
        if kind == self.tr("unit-step"):
            x = self.input_signals[self.tr("unit-step")](t, X0, d, A, w, phi)
            return x
        if kind == self.tr("impulse"):
            x = self.input_signals[self.tr("impulse")](t, X0, d, A, w, phi)
            #x = x_impulse(t, X0, d, A, w, phi)
            return x
        if kind == self.tr("dumping"):
            x = self.input_signals[self.tr("dumping")](t, X0, d, A, w, phi)
            return x
        if kind == self.tr("harmonic"):
            x = self.input_signals[self.tr("harmonic")](t, X0, d, A, w, phi)
            return x
            #return A * np.sin(w * t + phi)
        if kind == self.tr("underdamped"):
            x = self.input_signals[self.tr("underdamped")](t, X0, d, A, w, phi)
            return x
        if kind == self.tr("triangle"):
            x = self.input_signals[self.tr("triangle")](t, X0, d, A, w, phi)
            return x
        if kind == self.tr("square"):
            x = self.input_signals[self.tr("square")](t, X0, d, A, w, phi)
            return x
        return np.zeros_like(t)
        '''

    # --- Power/Energy placeholders ---
    def get_power(self, x, y, t, t1, t2):
        mask = (t >= t1) & (t <= t2)
        if not np.any(mask):
            return 0.0
        x_A_est, x_T_est, x_phi_est =  get_Awfi(t, x)
        y_A_est, y_T_est, y_phi_est =  get_Awfi(t, y)
        P = 0.5*x_A_est*y_A_est*np.cos(y_phi_est-x_phi_est)
        return P

    def get_complex_power(self, x, y, t, t1, t2):
        if self.input_sig_var.get() != self.tr('harmonic'):
            return np.nan
        mask = (t >= t1) & (t <= t2)
        if not np.any(mask):
            return np.nan
        
        x_A_est, x_T_est, x_phi_est =  get_Awfi(t, x)
        y_A_est, y_T_est, y_phi_est =  get_Awfi(t, y)
        P = 0.5*x_A_est*y_A_est*np.cos(y_phi_est-x_phi_est)
        Q = 0.5*x_A_est*y_A_est*np.sin(y_phi_est-x_phi_est)

        return P + 1j*Q

    def get_energy(self, x, y, t, t1, t2):
        mask = (t >= t1) & (t <= t2)
        if not np.any(mask):
            return 0.0
        return np.trapezoid(x[mask] * y[mask], t[mask])

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
        length=260
        )
        slider.pack(padx=10, pady=5)
        return slider
    
    def update_plot(self):
        Tmin = float(self.tmin_var.get())
        Tmax = float(self.tmax_var.get())
        dt = float(self.dt_var.get())
        if dt <= 0: dt = 1e-3
        t = np.arange(Tmin, Tmax+1e-12, dt)

        x = self.make_input_signal(self.input_sig_var.get(), t,
                              X0=self.X0_var.get(), d=self.d_var.get(), A=self.A_var.get(),
                              w=self.w_var.get(), phi=self.phi_var.get())
        params = {"R": self.R_var.get(), "L": self.L_var.get(), "C": self.C_var.get(), 
                  "C_u0": self.C_u0_var.get(), "L_i0": self.L_i0_var.get()}
        y = self.systems[self.system_var.get()](t, x, params)

        # t1, t2
        t1 = -np.inf if self.t1_choice.get() == "-inf" else self.t1_var.get()
        t2 = np.inf if self.t2_choice.get() == "inf" else self.t2_var.get()

        # Compute power & energy
        p = x * y
        if np.isfinite(t1) and np.isfinite(t2):
            xW = self.get_energy(x, x, t, t1, t2)
            yW = self.get_energy(y, y, t, t1, t2)
            xyW = self.get_energy(x, y, t, t1, t2)

            if self.input_sig_var.get() == self.tr('harmonic'):
                xP = self.get_power(x, x, t, t1, t2)
                xS = self.get_complex_power(x, x, t, t1, t2)
                yP = self.get_power(y, y, t, t1, t2)
                yS = self.get_complex_power(y, y, t, t1, t2)
                xyP = self.get_power(x, y, t, t1, t2)
                xyS = self.get_complex_power(x, y, t, t1, t2)
            else:
                xP = yP = xyP = 0.0
                xS = yS = xyS = 0.0
        else:
            xP = xW = xS = 0.0
            yP = yW = yS = 0.0
            xyP = xyW = xyS = 0.0

        # Update plots
        self.ax1.cla(); self.ax2.cla()
        self.ax1.plot(t, x, label="x(t)")
        self.ax1.plot(t, y, label="y(t)")
        if np.isfinite(t1): self.ax1.axvline(t1, color='r', linestyle='--')
        if np.isfinite(t2): self.ax1.axvline(t2, color='r', linestyle='--')
        self.ax1.set_xlabel("t [s]"); self.ax1.set_ylabel("x(t), y(t)"); self.ax1.grid(True); self.ax1.legend()
        #self.ax1.set_ylim([-10, 10]) # Put on slider?

        self.ax2.plot(t, p, label="p(t)=x(t) y(t)")
        if np.isfinite(t1): self.ax2.axvline(t1, color='r', linestyle='--')
        if np.isfinite(t2): self.ax2.axvline(t2, color='r', linestyle='--')
        self.ax2.set_xlabel("t [s]"); self.ax2.set_ylabel(self.tr("Power p(t)")); self.ax2.grid(True); self.ax2.legend()

        self.canvas.draw_idle()

        self.label_x_P.config(text=f"P = {xP:.2f}")
        self.label_x_W.config(text=f"W = {xW:.2f}")
        self.label_x_S.config(text=f"S = {xS:.2f}")

        self.label_y_P.config(text=f"P = {yP:.2f}")
        self.label_y_W.config(text=f"W = {yW:.2f}")
        self.label_y_S.config(text=f"S = {yS:.2f}")

        self.label_xy_P.config(text=f"P = {xyP:.2f}")
        self.label_xy_W.config(text=f"W = {xyW:.2f}")
        self.label_xy_S.config(text=f"S = {xyS:.2f}")

    def quit_app(self):
        self.root.destroy()
        