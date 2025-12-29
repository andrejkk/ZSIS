import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from signals_definitions import get_signals
from systemsTime_definitions import get_systems
import translations_Engl_Slo

# ----------------------
# Example system response functions
# ----------------------
def sys1_response_step(t, x, params):
    
    R = params['R']; L = params['L']; C = params['C'] #xi = params['xi']; 
    uC0 = params["C_u0"]; iL0 = params["L_i0"]
    X0 = params['X0']
    R = float(R); L = float(L); C = float(C) #xi = float(xi)
    X0 = float(X0)

    omega0 = 1 / np.sqrt(L * C)
    alpha = R / (2 * L)
    disc = alpha**2 - omega0**2
    if disc > 0:
        s1 = -alpha + np.sqrt(disc)
        s2 = -alpha - np.sqrt(disc)
        A = s2 / (s2 - s1)
        B = -s1 / (s2 - s1)
        return 1 - A * np.exp(s1 * t) - B * np.exp(s2 * t)
    elif disc == 0:
        s = -alpha
        return 1 - (1 + s * t) * np.exp(s * t)
    else:
        omega_d = np.sqrt(omega0**2 - alpha**2)
        return 1 - np.exp(-alpha * t) * (np.cos(omega_d * t) + alpha / omega_d * np.sin(omega_d * t))
    


def sys2_response_step(t, x, params):

    R = params['R']; L = params['L']; C = params['C'] #xi = params['xi']; 
    uC0 = params["C_u0"]; iL0 = params["L_i0"]
    X0 = params['X0']
    R = float(R); L = float(L); C = float(C) #xi = float(xi)
    X0 = float(X0)

    omega0 = 1 / np.sqrt(L * C) if C*L > 0 else np.inf
    alpha = R / (2 * L) if L > 0 else np.inf
    omega_d = np.sqrt(max(omega0**2 - alpha**2, 0))
    return np.exp(-alpha * t) * np.sin(omega_d * t)

def sys_zeros(params):

    R = params['R']; L = params['L']; C = params['C']    

    a = 1
    b = R / L
    c = 1 / (L * C)
    return np.roots([a, b, c])


# ----------------------
# GUI class
# ----------------------
class RLCApp:
    def __init__(self, root):
        self.root = root
        root.geometry("1100x800")
        
        self.lang = tk.StringVar(value="sl")
        root.title(self.tr("RLC System Response Explorer"))

        # Vars
        self.system_var = tk.StringVar(value=self.tr("Circuit 1"))
        self.R_var = tk.DoubleVar(value=0.0)
        self.L_var = tk.DoubleVar(value=1.0)
        self.C_var = tk.DoubleVar(value=0.2)
        self.C_u0_var = tk.DoubleVar(value=0.0)
        self.L_i0_var = tk.DoubleVar(value=0.0)
        self.X0_var = tk.DoubleVar(value=1.0)

        self.trace_real, self.trace_imag = [], []

        # --- LEFT PANEL with scroll ---
        left_container = tk.Frame(root)
        left_container.pack(side="left", padx=10, pady=10, fill="y")

        left_canvas = tk.Canvas(left_container)
        scrollbar = ttk.Scrollbar(left_container, orient="vertical",
                                  command=left_canvas.yview)
        scrollable = tk.Frame(left_canvas)

        scrollable.bind("<Configure>",
                        lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        left_canvas.create_window((0, 0), window=scrollable, anchor="nw")
        left_canvas.configure(yscrollcommand=scrollbar.set)

        left_canvas.pack(side="left", fill="y", expand=True)
        scrollbar.pack(side="right", fill="y")

        controls = scrollable

        # --- Controls ---
        #ttk.Label(controls, text="System:", font=("Helvetica", 12)).pack(pady=2)
        ttk.Label(controls, text=self.tr("System")).pack(pady=2)
        systems = [self.tr("Circuit 1"), self.tr("Circuit 2")]
        ttk.OptionMenu(controls, self.system_var, systems[0], *systems,
                       command=lambda *_: self.update_plot()).pack(fill="x", padx=4, pady=4)

        self.create_slider(controls, "R [Ω]", 0, 10, self.R_var, 0.01, 2.0)
        self.create_slider(controls, "L [H]", 0.1, 10, self.L_var, 0.01, 2.0)
        self.create_slider(controls, "C [F]", 0.1, 10, self.C_var, 0.01, 2.0)
        self.create_slider(controls, self.tr("i_L(0) [A]"), -10.0, 10.0, self.L_i0_var, 0.1, 1.0)
        self.create_slider(controls, "C [F]", 0.0, 10.0, self.C_var, 0.1, 2.0)
        self.create_slider(controls, self.tr("u_C(0) [V]"), -10.0, 10.0, self.C_u0_var, 0.1, 1.0)

        # --- RIGHT PANEL (plots) ---
        right = tk.Frame(root)
        right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.fig = Figure(figsize=(6, 7), dpi=80)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.update_plot()

    # Language 
    def tr(self, key):
        return translations_Engl_Slo.translations[self.lang.get()].get(key, key)

    def create_slider(self, frame, label, from_, to, var, step, tick):
        box = ttk.LabelFrame(frame, text=label, padding=5)
        box.pack(fill="x", pady=5)
        slider = tk.Scale(box, from_=from_, to=to, orient="horizontal",
                          variable=var, resolution=step, tickinterval=tick,
                          length=260, command=lambda *_: self.update_plot())
        slider.pack(padx=10, pady=5)
        return slider

    def update_plot(self):
        R, L, C, X0 = self.R_var.get(), self.L_var.get(), self.C_var.get(), self.X0_var.get()
        t = np.linspace(0, 10, 500)

        uC0 =self.C_u0_var.get(); iL0 = self.L_i0_var.get()
        params = { 'R': self.R_var.get(), 'L': self.L_var.get(), 'C': self.C_var.get(), 'X0': self.X0_var.get(),  "C_u0": uC0, "L_i0": iL0 }

        x = []
        if self.system_var.get() == self.tr("Circuit 1"):
            y = sys1_response_step(t, x, params)
        else:
            y = sys2_response_step(t, x, params)

        # poles
        roots = sys_zeros(params)
        self.trace_real.extend(np.real(roots))
        self.trace_imag.extend(np.imag(roots))

        # Plot 1
        self.ax1.cla()
        self.ax1.axhline(0, color="black", lw=1)
        self.ax1.axvline(0, color="black", lw=1)
        self.ax1.scatter(np.real(roots), np.imag(roots), c="red", s=80, label=self.tr("Current Poles"))
        self.ax1.plot(self.trace_real, self.trace_imag, "b--", alpha=0.6, label=self.tr("Pole Trace"))
        self.ax1.set_title(self.tr("Complex Plane")) #, fontsize=12)
        self.ax1.set_xlabel("Re"); self.ax1.set_ylabel("Im")
        self.ax1.set_xlim(-5, 5); self.ax1.set_ylim(-3, 3)
        self.ax1.grid(True); self.ax1.legend()

        # Plot 2
        self.ax2.cla()
        self.ax2.plot(t, y, label="y(t)")
        self.ax2.set_title(self.tr("System response")) # , fontsize=12)
        self.ax2.set_xlabel("Time [s]"); self.ax2.set_ylabel("y(t)")
        self.ax2.grid(True); self.ax2.legend()

        self.fig.tight_layout(pad=3.0) 
        self.canvas.draw_idle()

    def quit_app(self):
        self.root.destroy()


#if __name__ == "__main__":
#    root = tk.Tk()
#    app = RLCApp(root)
#    root.mainloop()
