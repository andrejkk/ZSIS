import tkinter as tk
from tkinter import ttk
import numpy as np
import math
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from systemsH_definitions import get_systems
import translations_Engl_Slo

"""
Resonance & Quality Interactive Simulator

REQUIREMENTS (you must provide these functions in the same namespace/module):
- sys_1_ZN_s(s, params) -> complex impedance Z(w)
- sys_1_YN_s(s, params) -> complex admittance Y(w)
- sys_1_sumP_s(s, params) -> float (sum of real powers at given w)
- sys_1_sumWC_s(s, params) -> float (sum of maximal capacitor energies at given w)
- sys_1_sumWL_s(s, params) -> float (sum of maximal inductor energies at given w)
- sys_1_Q() -> float (quality Q for system 1)
(and similarly for sys_2_* and sys_3_*)

If a function is missing, the GUI will print an error message in place of the corresponding value.
"""

def sys_1_ZN_s(s, params):
    R = float(params['R']); L = float(params['L']); C = float(params['C'])
    s = complex(s)
    ZN = R + s*L + 1.0 / (s*C) if np.abs(s*C)>0 else np.inf
    return ZN

def sys_1_YN_s(s, params):
    ZN = sys_1_ZN_s(s, params)
    ZN = np.asanyarray(ZN)
    YN = np.where(ZN != 0, 1.0 / ZN, np.inf)
    return YN 

def sys_1_sumP_s(s, params):
    I = float(1.0)
    R = float(params['R']); L = float(params['L']); C = float(params['C'])
    s = complex(s)
    return 0.5*R*(np.abs(I)**2) 

def sys_1_sumWL_s(s, params):
    I = 1
    R = float(params['R']); L = float(params['L']); C = float(params['C'])
    s = complex(s)
    return 0.5*L*(np.abs(I)**2)

def sys_1_sumWC_s(s, params):
    I = 1.0
    R = float(params['R']); L = float(params['L']); C = float(params['C'])
    s = complex(s)
    sumWC = 0.5*(1.0/(-s*s*C))*(np.abs(I)**2) if np.abs(s*C)>0 else np.inf
    return sumWC

def sys_1_Q_s(s, params):
    sumP = sys_1_sumP_s(s, params)
    sumWL = sys_1_sumWL_s(s, params)
    sumWC = sys_1_sumWC_s(s, params)
    Q = (s/1j) * sumWL / sumP if sumP > 0 else np.inf
    return Q



# Serial L and parallel R, C 
def sys_2_ZN_s(s, params):
    R = float(params['R']); L = float(params['L']); C = float(params['C'])
    s = complex(s)
    ZN = s*L + R / (1 + s*R*C)  if s*R*C!=0 else np.inf
    return ZN

def sys_2_YN_s(s, params):
    ZN = sys_2_ZN_s(s, params)
    ZN = np.asanyarray(ZN)
    YN = np.where(ZN != 0, 1.0 / ZN, np.inf) 
    return YN

def sys_2_sumP_s(s, params):
    I = 1.0
    R = float(params['R']); L = float(params['L']); C = float(params['C'])
    s = complex(s)
    return 0.5*R*(np.abs(I)**2)*(1.0 / (1 + (s*R*C/1j)**2))

def sys_2_sumWL_s(s, params):
    I = 1.0
    R = float(params['R']); L = float(params['L']); C = float(params['C'])
    s = complex(s)
    return 0.5*L*(np.abs(I)**2)

def sys_2_sumWC_s(s, params):
    I = 1.0
    R = float(params['R']); L = float(params['L']); C = float(params['C'])
    s = complex(s)
    return 0.5*C*(np.abs(I)**2)*C*(s*R*C/1j)**2 / (1 + (s*R*C/1j)**2)

def sys_2_Q_s(s, params):
    sumP = sys_2_sumP_s(s, params)
    sumWL = sys_2_sumWL_s(s, params)
    sumWC = sys_2_sumWC_s(s, params)
    Q = (s/1j) * sumWL / sumP if sumP > 0 else np.inf
    return Q

# Paralel: C and R + L
def sys_3_YN_s(s, params):
    R = float(params['R']); L = float(params['L']); C = float(params['C'])
    s = complex(s)
    YN = s*C + 1.0 / (R + s*L)
    return YN

def sys_3_ZN_s(s, params):
    YN = sys_3_YN_s(s, params)
    YN = np.asanyarray(YN)
    ZN = np.where(YN != 0, 1.0 / YN, np.inf) 
    return ZN

def sys_3_sumP_s(s, params):
    U = 1.0
    R = float(params['R']); L = float(params['L']); C = float(params['C'])
    s = complex(s)
    return 0.5*R*(np.abs(U)**2)*(1.0 / (R**2 + (s*L/1j)**2))

def sys_3_sumWL_s(s, params):
    U = 1
    R = float(params['R']); L = float(params['L']); C = float(params['C'])
    s = complex(s)
    return 0.5*L*(np.abs(U)**2)*(1.0 / (R**2 + (s*L/1j)**2))

def sys_3_sumWC_s(s, params):
    U = 1
    R = float(params['R']); L = float(params['L']); C = float(params['C'])
    s = complex(s)
    return 0.5*C*((np.abs(U)**2))

def sys_3_Q_s(s, params):
    sumP = sys_3_sumP_s(s, params)
    sumWL = sys_3_sumWL_s(s, params)
    sumWC = sys_3_sumWC_s(s, params)
    Q = (s/1j) * sumWL / sumP if sumP > 0 else np.inf
    return Q



class ResonanceQualitySimulator:
    def __init__(self, root):
        self.root = root
        root.geometry("1100x700")
        self.lang = tk.StringVar(value="sl")
        root.title(self.tr("Resonance & Quality Simulator"))


        self.systems = {self.tr("Circuit 1"): {"ZN_fun": sys_1_ZN_s, 
                                      "YN_fun": sys_1_YN_s,
                                      "sumP_fun": sys_1_sumP_s,
                                      "sumWL_fun": sys_1_sumWL_s,
                                      "sumWC_fun": sys_1_sumWC_s,
                                      "Q_fun": sys_1_Q_s, 
                                      "image": "sys_1_netFigure.png"},
                        self.tr("Circuit 2"): {"ZN_fun": sys_2_ZN_s, 
                                      "YN_fun": sys_2_YN_s,
                                      "sumP_fun": sys_2_sumP_s,
                                      "sumWL_fun": sys_2_sumWL_s,
                                      "sumWC_fun": sys_2_sumWC_s,
                                      "Q_fun": sys_2_Q_s, 
                                      "image": "sys_2_netFigure.png"},
                        self.tr("Circuit 3"): {"ZN_fun": sys_3_ZN_s, 
                                      "YN_fun": sys_3_YN_s,
                                      "sumP_fun": sys_3_sumP_s,
                                      "sumWL_fun": sys_3_sumWL_s,
                                      "sumWC_fun": sys_3_sumWC_s,
                                      "Q_fun": sys_3_Q_s, 
                                      "image": "sys_3_netFigure.png"},
                        }
        
        # Variables
        self.CIRCUIT_NAMES = list(self.systems.keys())
        self.system_var = tk.StringVar(value=self.CIRCUIT_NAMES[0])
        self.resonance_type = tk.StringVar(value="voltage")  # 'voltage' or 'current'
        self.w_var = tk.DoubleVar(value=2.0)   # current angular frequency for phasor plot
        self.R_var = tk.DoubleVar(value=1.0)
        self.L_var = tk.DoubleVar(value=1.0)
        self.C_var = tk.DoubleVar(value=1.0)

        # Frequency sweep range controls
        self.wmin_var = tk.DoubleVar(value=0.01)
        self.wmax_var = tk.DoubleVar(value=10.0)
        self.wpoints_var = tk.IntVar(value=400)

        # Left control panel with vertical scrollbar
        left_container = tk.Frame(root)
        left_container.pack(side="left", fill="y", padx=5, pady=5)

        left_canvas = tk.Canvas(left_container, width=360)  # adjust width for sliders
        left_scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=left_canvas.yview)
        self.left_frame = ttk.Frame(left_canvas)

        self.left_frame.bind(
            "<Configure>",
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        )

        left_canvas.create_window((0, 0), window=self.left_frame, anchor="nw")
        left_canvas.configure(yscrollcommand=left_scrollbar.set)


        left_canvas.pack(side="left", fill="y", expand=True)
        left_scrollbar.pack(side="right", fill="y")

        # Now put all the controls into self.left_frame instead of 'left'
        f_sys = ttk.LabelFrame(self.left_frame, text=self.tr("select_system"), padding=6)
        f_sys.pack(fill="x", pady=4)
        ttk.OptionMenu(f_sys, self.system_var, self.CIRCUIT_NAMES[0], *self.CIRCUIT_NAMES).pack(fill="x", padx=4, pady=4)

        f_res = ttk.LabelFrame(self.left_frame, text=self.tr("Resonance type"), padding=6)
        f_res.pack(fill="x", pady=4)
        ttk.OptionMenu(f_res, self.resonance_type, self.tr("voltage"), self.tr("voltage"), self.tr("current")).pack(fill="x", padx=4, pady=4)

        f_rlc = ttk.LabelFrame(self.left_frame, text=self.tr("system_elements"), padding=6)
        f_rlc.pack(fill="x", pady=4)
        self.create_slider(f_rlc, "R [Ω]", 0.0, 50.0, self.R_var, 0.01, tick_interval=5.0)
        self.create_slider(f_rlc, "L [H]", 0.0, 50.0, self.L_var, 0.001, tick_interval=5.0)
        self.create_slider(f_rlc, "C [F]", 0.0, 50.0, self.C_var, 0.001, tick_interval=5.0)

        f_w = ttk.LabelFrame(self.left_frame, text=self.tr("Angular frequency ω"), padding=6)
        f_w.pack(fill="x", pady=4)
        self.create_slider(f_w, "ω [rad/s]", 0.0, 50.0, self.w_var, 0.001, tick_interval=5.0)
        self.create_slider(f_w, self.tr("ω min"), 0.001, 50.0, self.wmin_var, 0.001, tick_interval=5.0)
        self.create_slider(f_w, self.tr("ω max"), 0.01, 200.0, self.wmax_var, 0.1, tick_interval=20.0)
        self.create_slider(f_w, self.tr("Sweep points"), 50, 2000, self.wpoints_var, 1, tick_interval=200)
        

        
        # Right frame: plots + info
        right = ttk.Frame(root)
        right.pack(side="right", fill="both", expand=True)

        self.fig = Figure(figsize=(8,6), dpi=100)
        # Top: complex plane
        self.ax_complex = self.fig.add_subplot(211)
        self.ax_complex.set_title(self.tr("Complex plane: phasors U (voltage) and I (current)"))
        self.ax_complex.set_xlabel("Re")
        self.ax_complex.set_ylabel("Img")
        self.ax_complex.grid(True)
        self.ax_complex.set_aspect('equal', adjustable='box')

        # Bottom: magnitude vs frequency
        self.ax_mag = self.fig.add_subplot(212)
        self.ax_mag.set_title(self.tr("Magnitude vs ω (marker = current ω)"))
        self.ax_mag.set_xlabel("ω [rad/s]")
        self.ax_mag.set_ylabel(self.tr("|U| or |I|"))
        self.ax_mag.grid(True)
        self.fig.subplots_adjust(hspace=0.4)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Info labels under the canvas
        info_frame = ttk.Frame(right)
        info_frame.pack(fill="x", pady=6)
        self.label_sumP = ttk.Label(info_frame, text="sum P:")
        self.label_sumP.pack(anchor="w")
        self.label_sumWC = ttk.Label(info_frame, text="sum W_C:")
        self.label_sumWC.pack(anchor="w")
        self.label_sumWL = ttk.Label(info_frame, text="sum W_L:")
        self.label_sumWL.pack(anchor="w")
        self.label_Q = ttk.Label(info_frame, text="Quality Q:")
        self.label_Q.pack(anchor="w")

        # Trace variable changes to update
        for v in [self.system_var, self.resonance_type, self.w_var, self.R_var, self.L_var, self.C_var,
                  self.wmin_var, self.wmax_var, self.wpoints_var]:
            v.trace_add("write", lambda *a: self.update_all())

        # initial draw
        self.update_all()
        
    # Language 
    def tr(self, key):
        return translations_Engl_Slo.translations[self.lang.get()].get(key, key)
        

    def create_slider(self, parent, label_text, frm, to, var, resolution, tick_interval=0.0):
        f = ttk.LabelFrame(parent, text=label_text, padding=4)
        f.pack(fill="x", pady=4)
        # tk.Scale doesn't accept IntVar with large range reliably; we'll use Double and cast when needed
        slider = tk.Scale(f, from_=frm, to=to, orient="horizontal", variable=var,
                          resolution=resolution, length=300, tickinterval=tick_interval)
        slider.pack(padx=6, pady=4)
        return slider


    '''
    def _call_fn(self, fn_name, *args, default=None):
        """Try to call function named fn_name in globals(); return default on failure and print error."""
        g = globals()
        if fn_name in g and callable(g[fn_name]):
            try:
                return g[fn_name](*args)
            except Exception as e:
                # return default and embed error message in the label prints
                return default if default is not None else None
        else:
            return default if default is not None else None
    '''

    def _get_index_for_system(self):
        name = self.system_var.get()
        if name == self.tr("Circuit 1"): return 1
        if name == self.tr("Circuit 2"): return 2
        if name == self.tr("Circuit 3"): return 3
        # fallback
        return 1

    def update_all(self):
        # read parameters
        params = {"R": float(self.R_var.get()), "L": float(self.L_var.get()), "C": float(self.C_var.get())}
        w = float(self.w_var.get())
        wmin = float(self.wmin_var.get())
        wmax = float(self.wmax_var.get())
        wpoints = int(round(self.wpoints_var.get()))
        if wpoints < 10: wpoints = 10


        sys_name = self.system_var.get()
        Z_fn_name = self.systems[sys_name]['ZN_fun']
        Y_fn_name = self.systems[sys_name]['YN_fun']
        sumP_fn_name = self.systems[sys_name]['sumP_fun']
        sumWL_fn_name = self.systems[sys_name]['sumWL_fun']
        sumWC_fn_name = self.systems[sys_name]['sumWC_fun']
        Q_fn_name = self.systems[sys_name]['Q_fun']

        # idx = self._get_index_for_system()
        #Z_fn_name = f"sys_{idx}_ZN"
        #Y_fn_name = f"sys_{idx}_YN"
        #sumP_fn_name = f"sys_{idx}_sumP"
        #sumWC_fn_name = f"sys_{idx}_sumWC"
        #sumWL_fn_name = f"sys_{idx}_sumWL"
        #Q_fn_name = f"sys_{idx}_Q"

        # compute Z and Y at current w (if possible)
        s=1j*w
        Z = Z_fn_name(s, params) 
        Y = Y_fn_name(s, params)
        #print (s, Z, Y)

        # Determine phasors based on resonance selection
        res_type = self.resonance_type.get().lower()
        # convention: phasors are complex numbers (real reference)
        U = None; I = None
        if res_type == "voltage":
            # I is assumed 1 (phasor 1+0j)
            I = 1.0 + 0j
            if Z is not None:
                U = Z * I
            else:
                U = None
        else:  # current resonance
            U = 1.0 + 0j
            if Y is not None:
                I = Y * U
            else:
                I = None

        # --- Complex plane plot ---
        self.ax_complex.cla()
        self.ax_complex.set_title(self.tr("Complex plane: phasors U (voltage) and I (current)"))
        self.ax_complex.set_xlabel("Re")
        self.ax_complex.set_ylabel("Im")
        self.ax_complex.grid(True)
        self.ax_complex.set_aspect('equal', adjustable='box')

        # Plot phasor arrows if available; if not available, show text message
        phasor_vals = []
        phasor_labels = []
        if U is not None:
            phasor_vals.append(U)
            phasor_labels.append("U")
        if I is not None:
            phasor_vals.append(I)
            phasor_labels.append("I")

        if phasor_vals:
            reals = [p.real for p in phasor_vals]
            imags = [p.imag for p in phasor_vals]
            # determine axis limits with padding
            max_abs = max([abs(x) for x in reals + imags] + [1e-6])
            pad = 0.2 * max_abs
            xlim = (-max_abs - pad, max_abs + pad)
            ylim = (-max_abs - pad, max_abs + pad)
            # expand if any phasor has larger separate real/imag
            mag_vals = [abs(p) for p in phasor_vals]
            max_mag = max(mag_vals) if mag_vals else 1.0
            # set symmetric axes centered at zero
            axis_limit = max(max(abs(xlim[0]), abs(xlim[1])), max(abs(ylim[0]), abs(ylim[1])), max_mag + pad)
            self.ax_complex.set_xlim(-axis_limit, axis_limit)
            self.ax_complex.set_ylim(-axis_limit, axis_limit)

            # plot arrows
            colors = {"U": "tab:blue", "I": "tab:orange"}
            for p, lab in zip(phasor_vals, phasor_labels):
                # arrow from origin to point p
                self.ax_complex.arrow(0, 0, p.real, p.imag, head_width=0.05*axis_limit,
                                      head_length=0.08*axis_limit, length_includes_head=True,
                                      fc=colors.get(lab, "black"), ec=colors.get(lab, "black"))
                self.ax_complex.text(p.real * 1.05, p.imag * 1.05, f"{lab} ({p.real:.3g}+j{p.imag:.3g})")
        else:
            self.ax_complex.text(0.5, 0.5, "Impedance/admittance function(s) missing\nCannot compute phasors",
                                 transform=self.ax_complex.transAxes, ha='center')

        # --- Magnitude vs omega plot ---
        self.ax_mag.cla()
        self.ax_mag.set_title(self.tr("Magnitude vs ω (marker = current ω)"))
        self.ax_mag.set_xlabel("ω [rad/s]")
        self.ax_mag.set_ylabel(self.tr("|U| (voltage-res) or |I| (current-res)"))
        self.ax_mag.grid(True)

        # generate sweep
        if wmax <= wmin:
            wmax = wmin + 1e-3
        # choose linear sweep (user can choose range)
        w_list = np.linspace(wmin, wmax, wpoints)

        mags = np.full_like(w_list, np.nan, dtype=float)
        for i_w, w_s in enumerate(w_list):
            s_s = 1j*w_s
            Z_s = Z_fn_name(s_s, params) 
            Y_s = Y_fn_name(s_s, params)
            if res_type == self.tr("voltage"):
                # I assumed 1 => |U| = |Z|*|I| = |Z|
                if Z_s is not None:
                    mags[i_w] = abs(Z_s * (1.0 + 0j))
            if res_type == self.tr("current"):
                # U assumed 1 => |I| = |Y|*|U| = |Y|
                if Y_s is not None:
                    mags[i_w] = abs(Y_s * (1.0 + 0j))

        # plot magnitude sweep (skip nan points)
        valid = ~np.isnan(mags)
        if np.any(valid):
            self.ax_mag.plot(w_list[valid], mags[valid], label=("|U|" if res_type==self.tr("voltage") else "|I|"))
            # vertical line at current w
            self.ax_mag.axvline(w, color='r', linestyle='--', label=f"ω = {w:.3g}")
            # highlight the current magnitude
            # compute current point magnitude
            current_mag = None
            if res_type == self.tr("voltage") and Z is not None:
                current_mag = abs(Z)
            if res_type == self.tr("current") and Y is not None:
                current_mag = abs(Y)
            if current_mag is not None:
                self.ax_mag.scatter([w], [current_mag], color='red', zorder=5)
            self.ax_mag.legend()
        else:
            self.ax_mag.text(0.5, 0.5, "Cannot compute magnitude sweep.\nImpedance/admittance missing.",
                             transform=self.ax_mag.transAxes, ha='center')

        self.canvas.draw_idle()

        # --- Info outputs: call sumP, sumWC, sumWL and Q functions ---
        s = 1j*w
        sumP = sumP_fn_name(s, params)
        sumWC = sumWC_fn_name(s, params)
        sumWL = sumWL_fn_name(s, params)
        Qval = Q_fn_name(s, params)

        

        self.label_sumP.config(text=f"sum P (@ ω={w:.6g}) = {sumP:.4g}")
        self.label_sumWC.config(text=f"sum W_C (@ ω={w:.6g}) = {sumWC:.4g}")
        self.label_sumWL.config(text=f"sum W_L (@ ω={w:.6g}) = {sumWL:.4g}")
        self.label_Q.config(text=self.tr("Quality Q") + f"(@ ω={w:.6g}) = {Qval:.4g}")


