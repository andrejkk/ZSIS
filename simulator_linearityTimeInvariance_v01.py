import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import style
from signals_definitions import get_signals
from systemsTime_definitions import get_systems
import translations_Engl_Slo

style.use("seaborn-v0_8")

'''
# --- System functions ---
def sys_response_1(t, x, params):
    return params['R'] * x

def sys_response_2(t, x, params):
    R = params['R']
    al = 0.1
    return R * (1 + al * t) * x

def system_1N4148(t, u, params):
    I_s = 2e-9
    n = 1.7
    V_T = 25.85e-3
    return I_s*(np.exp(u/(n*V_T)) - 1)

def sys_response_3(t, x, params):
    return x**2

def sys_response_4(t, x, params):
    return np.tanh(x)



# --- Signals ---
def x_constant(t, X0, freq_cos, phase_cos, freq_square, slope_ramp):
    return X0 * np.ones_like(t)

def x_cosine(t, X0, freq_cos, phase_cos, freq_square, slope_ramp):
    return np.cos(2*np.pi*freq_cos*t + phase_cos)

def x_square(t, X0, freq_cos, phase_cos, freq_square, slope_ramp):
    return np.sign(np.sin(2*np.pi*freq_square*t))

def x_unitStep(t, X0, freq_cos, phase_cos, freq_square, slope_ramp):
    return slope_ramp * t
'''

# t = np.linspace(0,10,500)

# ===================================================
class SystemPropertiesApp:
    def __init__(self, root):
        self.root = root
        self.lang = tk.StringVar(value="sl")
        self.root.title(self.tr("System Linearity Simulation"))

        self.input_signals = get_signals(self.tr)
        self.systems = get_systems(self.tr)

        '''
        self.signals = {
            self.tr("Constant"): x_constant,
            self.tr("Cosine"): x_cosine,
            self.tr("Square"): x_square,
            self.tr("Unit-step"): x_unitStep
        }

        self.systems = {
            "Resistor - R": sys_response_1,
            "Resistor - R(T)": sys_response_2,
            "Diode - nonlinear char.": system_1N4148,
            "Nonlinear: y = x^2": sys_response_3,
            "Saturating: y = tanh(x)": sys_response_4
        }
        '''

        # Variables
        self.selected_system = tk.StringVar(value=list(self.systems.keys())[0])
        self.selected_signal1 = tk.StringVar(value=self.tr("Cosine"))
        self.selected_signal2 = tk.StringVar(value=self.tr("Square"))

        self.t = np.linspace(0,10,500)

        self.X0_1 = tk.DoubleVar(value=1.0)
        self.cos_freq1 = tk.DoubleVar(value=1.0)
        self.cos_phase1 = tk.DoubleVar(value=0.0)
        self.square_freq1 = tk.DoubleVar(value=1.0)
        self.ramp_slope1 = tk.DoubleVar(value=0.5)

        self.X0_2 = tk.DoubleVar(value=1.0)
        self.cos_freq2 = tk.DoubleVar(value=1.0)
        self.cos_phase2 = tk.DoubleVar(value=0.0)
        self.square_freq2 = tk.DoubleVar(value=1.0)
        self.ramp_slope2 = tk.DoubleVar(value=0.5)

        self.R_var = tk.DoubleVar(value=1.0)
        self.L_var = tk.DoubleVar(value=1.0)
        self.C_var = tk.DoubleVar(value=1.0)

        self.a1 = tk.DoubleVar(value=1.0)
        self.a2 = tk.DoubleVar(value=1.0)

        # Build UI
        self.build_frames()
        self.build_sliders()
        self.build_plots()
        self.build_time_slider()
        self.bind_updates()
        self.update_plot()
       



    # ------------------ scrollable frames ------------------
    def build_frames(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left scrollable
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        left_canvas = tk.Canvas(left_frame, width=350) # 350
        left_canvas.pack(side=tk.LEFT, fill=tk.Y, expand=True)

        scrollbar_left = ttk.Scrollbar(left_frame, orient="vertical", command=left_canvas.yview)
        scrollbar_left.pack(side=tk.RIGHT, fill=tk.Y)

        self.left_scrollable = tk.Frame(left_canvas)
        self.left_scrollable.bind("<Configure>", lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        left_canvas.create_window((0,0), window=self.left_scrollable, anchor="nw")
        left_canvas.configure(yscrollcommand=scrollbar_left.set)

        # Right frame for plots
        self.right_frame = tk.Frame(main_frame)
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # ------------------ create slider with canvas ticks ------------------
    def create_slider(self, parent, label_text, var, from_, to_, resolution, length=250):
        frame = tk.Frame(parent)
        frame.pack(padx=10, pady=5, fill=tk.X)

        # Label
        tk.Label(frame, text=label_text).pack()

        # Slider
        scale = tk.Scale(frame, from_=from_, to=to_, orient=tk.HORIZONTAL,
                        resolution=resolution, variable=var, length=length, showvalue=True)
        scale.pack()

        # Canvas for ticks (same width as scale)
        tick_canvas = tk.Canvas(frame, height=35, width=length)
        tick_canvas.pack()

        num_ticks = 6
        # Tkinter knob has margins at both ends, ~15px each
        margin = 15
        usable_length = length - 2 * margin

        for i in range(num_ticks):
            val = from_ + i * (to_ - from_) / (num_ticks - 1)

            # Map value → usable range
            rel = (val - from_) / (to_ - from_)
            if from_ < to_:
                x_pos = margin + rel * usable_length
            else:
                x_pos = margin + (1 - rel) * usable_length

            # Tick line
            tick_canvas.create_line(x_pos, 0, x_pos, 8)

            # Tick label
            anchor = "n"
            if i == 0:
                anchor = "nw"
            elif i == num_ticks - 1:
                anchor = "ne"

            tick_canvas.create_text(x_pos, 20, text=f"{val:.2f}", anchor=anchor)

        return scale



    '''
    def create_slider(self, parent, label_text, var, from_, to_, resolution, length=250): 
        frame = tk.Frame(parent)
        frame.pack(padx=10, pady=5, fill=tk.X)

        tk.Label(frame, text=label_text).pack()
        scale = tk.Scale(frame, from_=from_, to=to_, orient=tk.HORIZONTAL, resolution=resolution,
                         variable=var, length=length)
        scale.pack()

        # Canvas for ticks
        tick_canvas = tk.Canvas(frame, height=32, width=length) # Was 
        tick_canvas.pack()
        num_ticks = 6
        for i in range(num_ticks):
            val = from_ + i*(to_-from_)/(num_ticks-1)
            x_pos = i*length/(num_ticks-1)
            tick_canvas.create_line(x_pos, 0, x_pos, 8)
            tick_canvas.create_text(x_pos, 20, text=f"{val:.2f}", anchor="n")
        return scale
    '''

    # ------------------ build sliders ------------------
    def build_sliders(self):
        # System selection
        sys_frame = tk.LabelFrame(self.left_scrollable, text="System Selection")
        sys_frame.pack(fill=tk.X, pady=5)
        ttk.Label(sys_frame, text="Select System:").pack()
        system_menu = ttk.Combobox(sys_frame, textvariable=self.selected_system,
                                   values=list(self.systems.keys()), state="readonly")
        system_menu.pack()

        # Signals
        sig_frame = tk.LabelFrame(self.left_scrollable, text=self.tr("Signals"))
        sig_frame.pack(fill=tk.X, pady=5)
        ttk.Label(sig_frame, text="Signal 1:").pack()
        sig_menu1 = ttk.Combobox(sig_frame, textvariable=self.selected_signal1,
                                 values=list(self.input_signals.keys()), state="readonly")
        sig_menu1.pack()
        ttk.Label(sig_frame, text="Signal 2:").pack()
        sig_menu2 = ttk.Combobox(sig_frame, textvariable=self.selected_signal2,
                                 values=list(self.input_signals.keys()), state="readonly")
        sig_menu2.pack()

        # Coefficients
        coeff_frame = tk.LabelFrame(self.left_scrollable, text=self.tr("Coefficients"))
        coeff_frame.pack(fill=tk.X, pady=5)
        self.create_slider(coeff_frame, "a1", self.a1, -2, 2, 0.1)
        self.create_slider(coeff_frame, "a2", self.a2, -2, 2, 0.1)

        # System parameters
        sys_par_frame = tk.LabelFrame(self.left_scrollable, text=self.tr("System Parameters"))
        sys_par_frame.pack(fill=tk.X, pady=5)
        self.create_slider(sys_par_frame, "R", self.R_var, 0.1, 5.0, 0.1)
        self.create_slider(sys_par_frame, "L", self.L_var, 0.1, 5.0, 0.1)
        self.create_slider(sys_par_frame, "C", self.C_var, 0.1, 5.0, 0.1)

        # Signal 1 sliders
        sig1_frame = tk.LabelFrame(self.left_scrollable, text=self.tr("Signal 1 Parameters"))
        sig1_frame.pack(fill=tk.X, pady=5)
        self.create_slider(sig1_frame, "X0", self.X0_1, 0.1, 5.0, 0.1)
        self.create_slider(sig1_frame, self.tr("Cos Freq"), self.cos_freq1, 0.1, 5.0, 0.1)
        self.create_slider(sig1_frame, self.tr("Cos Phase"), self.cos_phase1, -np.pi, np.pi, 0.1)
        self.create_slider(sig1_frame, self.tr("Square Freq"), self.square_freq1, 0.1, 5.0, 0.1)
        self.create_slider(sig1_frame, self.tr("Ramp Slope"), self.ramp_slope1, -2.0, 2.0, 0.1)

        # Signal 2 sliders
        sig2_frame = tk.LabelFrame(self.left_scrollable, text=self.tr("Signal 2 Parameters"))
        sig2_frame.pack(fill=tk.X, pady=5)
        self.create_slider(sig2_frame, "X0", self.X0_2, 0.1, 5.0, 0.1)
        self.create_slider(sig2_frame, self.tr("Cos Freq"), self.cos_freq2, 0.1, 5.0, 0.1)
        self.create_slider(sig2_frame, self.tr("Cos Phase"), self.cos_phase2, -np.pi, np.pi, 0.1)
        self.create_slider(sig2_frame, self.tr("Square Freq"), self.square_freq2, 0.1, 5.0, 0.1)
        self.create_slider(sig2_frame, self.tr("Ramp Slope"), self.ramp_slope2, -2.0, 2.0, 0.1)

    # ------------------ plots ------------------
    def build_plots(self):
        self.fig = Figure(figsize=(8,6), dpi=100)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.fig.tight_layout(pad=3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ------------------ time slider ------------------
    def build_time_slider(self):
        time_frame = tk.LabelFrame(self.root, text=self.tr("Select Time for Marker"))
        time_frame.pack(fill=tk.X, pady=10, ipady=10)
        self.time_slider = self.create_slider(time_frame, self.tr("Time [s]"), tk.DoubleVar(value=0),
                                              0, len(self.t)-1, 1, length=600)

    # ------------------ compute signal ------------------
    def compute_signal(self, signal_type, X0, freq_cos, phase_cos, freq_square, slope_ramp):
        return self.input_signals[signal_type](self.t, X0,freq_cos,phase_cos,freq_square,slope_ramp)
        #if signal_type == self.tr("Constant"): return self.input_signals[self.tr("Constant")](self.t, X0,freq_cos,phase_cos,freq_square,slope_ramp)
        #if signal_type == self.tr("Cosine"): return x_cosine(self.t, X0,freq_cos,phase_cos,freq_square,slope_ramp)
        #if signal_type == self.tr("Square"): return x_square(self.t, X0,freq_cos,phase_cos,freq_square,slope_ramp)
        #if signal_type == self.tr("Unit-step"): return x_unitStep(self.t, X0,freq_cos,phase_cos,freq_square,slope_ramp)
        #return np.zeros_like(self.t)

    # ------------------ update plot ------------------
    def update_plot(self, *args):
        self.ax1.clear()
        self.ax2.clear()

        x1 = self.compute_signal(self.selected_signal1.get(),
                                 self.X0_1.get(), self.cos_freq1.get(),
                                 self.cos_phase1.get(), self.square_freq1.get(),
                                 self.ramp_slope1.get())
        x2 = self.compute_signal(self.selected_signal2.get(),
                                 self.X0_2.get(), self.cos_freq2.get(),
                                 self.cos_phase2.get(), self.square_freq2.get(),
                                 self.ramp_slope2.get())
        x = self.a1.get()*x1 + self.a2.get()*x2

        params = {'R': self.R_var.get(), 'L': self.L_var.get(), 'C': self.C_var.get()}
        y = self.systems[self.selected_system.get()](self.t, x, params)

        self.ax1.plot(self.t, x, label="x(t) = a1*x1 + a2*x2")
        self.ax1.plot(self.t, y, label="y(t)")
        self.ax1.set_title(self.tr("Time Domain"))
        self.ax1.set_xlabel(self.tr("Time [s]"))
        self.ax1.set_ylabel(self.tr("Amplitude"))
        self.ax1.legend()
        self.ax1.grid(True)

        sx, sy = zip(*sorted(zip(x, y)))
        self.ax2.plot(sx, sy, color='orange')
        self.ax2.set_title("(x(t), y(t))")
        self.ax2.set_xlabel("x(t)")
        self.ax2.set_ylabel("y(t)")
        self.ax2.grid(True)

        idx = int(self.time_slider.get())
        if 0 <= idx < len(self.t):
            self.ax2.plot(x[idx], y[idx], 'ro')
            self.ax2.axhline(y[idx], color='red', linestyle='--')
            self.ax2.axvline(x[idx], color='red', linestyle='--')

        self.canvas.draw()
    
    # Language 
    def tr(self, key):
        return translations_Engl_Slo.translations[self.lang.get()].get(key, key)

    # ------------------ bind updates ------------------
    def bind_updates(self):
        for var in [self.X0_1, self.cos_freq1, self.cos_phase1, self.square_freq1, self.ramp_slope1,
                    self.X0_2, self.cos_freq2, self.cos_phase2, self.square_freq2, self.ramp_slope2,
                    self.a1, self.a2, self.R_var, self.L_var, self.C_var,
                    self.selected_system, self.selected_signal1, self.selected_signal2]:
            var.trace_add("write", lambda *a: self.update_plot())
        self.time_slider.config(command=lambda v: self.update_plot())

    def quit_app(self):
        self.root.destroy()

    