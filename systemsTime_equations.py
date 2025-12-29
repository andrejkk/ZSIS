import numpy as np

# Resistor
def sys_resistor_xi_yu_y(t, x, params):
    ''' Resistor i->u'''
    R = params['R']; L = params['L']; C = params['C']
    y = R * x
    return y[:len(t)]

def sys_resistor_xu_yi_y(t, x, params):
    ''' Resistor'''
    R = params['R']; L = params['L']; C = params['C']
    if R > 0:
        y = x / R
    else:
        y=np.nan*np.ones_like(t) 
    return y[:len(t)]

def sys_capacitor_xi_yu_y(t, x, params):
    ''' Capacitor i->u'''
    R = params['R']; L = params['L']; C = params['C']
    C_u0 = params['C_u0']

    dt = np.diff(t)
    increments = 0.5 * (x[1:] + x[:-1]) * dt  # trapezoidal rule
    y = C_u0 + (1/C) * np.concatenate(([0.0], np.cumsum(increments))) if C > 0 else np.inf * np.ones_like(t)
    
#    dt = np.mean(np.diff(t))
#    h = np.ones_like(t) / C
#    y = np.convolve(x, h) * dt
    y = y[:len(x)] 
    return y  

def sys_capacitor_xu_yi_y(t, x, params):
    ''' Capacitor '''
    R = params['R']; L = params['L']; C = params['C']
    C_u0 = params['C_u0']

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    dxdt = np.zeros_like(x) 
    dxdt[0] = (x[1] - x[0]) / (t[1] - t[0]) # first point
    dxdt[-1] = (x[-1] - x[-2]) / (t[-1] - t[-2]) # last point
    dxdt[1:-1] = (x[2:] - x[:-2]) / (t[2:] - t[:-2])
    y = C * dxdt

    return y  


def sys_inductor_xi_yu_y(t, x, params):
    ''' Coil - inductor '''
    R = params['R']; L = params['L']; C = params['C']
    L_i0 = params['L_i0']
    
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    
    dxdt = np.zeros_like(x) 
    dxdt[0] = (x[1] - x[0]) / (t[1] - t[0]) # first point
    dxdt[-1] = (x[-1] - x[-2]) / (t[-1] - t[-2]) # last point
    dxdt[1:-1] = (x[2:] - x[:-2]) / (t[2:] - t[:-2])
    y = L * dxdt

    return y[:len(t)]

def sys_inductor_xu_yi_y(t, x, params):
    ''' Coil - inductor '''
    R = params['R']; L = params['L']; C = params['C']
    L_i0 = params['L_i0']
    
    dt = np.diff(t)
    increments = 0.5 * (x[1:] + x[:-1]) * dt  
    y = L_i0 + (1/L) * np.concatenate(([0.0], np.cumsum(increments))) if L > 0 else np.inf * np.ones_like(t)
  
    y = L_i0 + y[:len(x)] 
    return y[:len(t)]

# Abstract system based on convolution
def sys_serialRC_xi_yu_y(t, x, params):
    R = params['R']; L = params['L']; C = params['C']
    C_u0 = params['C_u0']
    
    if C == 0:
        return np.zeros_like(t)
    
    dt = np.diff(t)
    increments = 0.5 * (x[1:] + x[:-1]) * dt  # trapezoidal rule
    y = R * x + C_u0 + (1/C) * np.concatenate(([0.0], np.cumsum(increments))) if C > 0 else np.inf * np.ones_like(t)
    
    return y[:len(t)]


def sys_resistor_RT_xi_yu_y(t, x, params):
    R = params['R']
    al = 0.1
    return R * (1 + al * t) * x

def system_1N4148(t, u, params):
    I_s = 2e-9
    n = 1.7
    V_T = 25.85e-3
    return I_s*(np.exp(u/(n*V_T)) - 1)

def sys_nonlinearSq(t, x, params):
    return x**2

def sys_saturating(t, x, params):
    return np.tanh(x)



def sys_CircuitRLC_xi_yu_y(t, x, params):
    ''' Circuit for transient responce'''
    R = params['R']; L = params['L']; C = params['C']

    if L * C > 0:
        D = R*R - 4*L/C
        if D >= 0:
            p1 = (-R + np.sqrt(D)) / (2*L)
            p2 = (-R - np.sqrt(D)) / (2*L)
        else:
            p1 = (-R + 1j*np.sqrt(-D)) / (2*L)
            p2 = (-R - 1j*np.sqrt(-D)) / (2*L)
    else:
        p1, p2 = np.nan, np.nan


    # Not a critical dumping
    Ig0 = 1
    if p1 - p2 != 0:
        y = ((R*Ig0)/(p1-p2))*(p2*np.exp(p1*t) - p1*np.exp(p2*t)) + R*Ig0
    else:
        y = R*Ig0*(-1+p1*t)*np.exp(p1*t) + R*Ig0

    return np.real(y)
    
    # EXAMPLE: simple RC first-order lowpass with tau = R*C
    #tau = R * C if (R*C) != 0 else 1e9
    # impulse response h(t) = (1/tau) exp(-t/tau)
    #dt = t[1] - t[0] if len(t)>1 else 1.0
    #h = (1.0/tau) * np.exp(-t / tau)
    #y = np.convolve(x, h) * dt
    #return y[: len(t)]



def sys_dumpingRC_xi_yu_y(t, x, params):

    R = params['R']; L = params['L']; C = params['C']
    y = np.exp(-t/(R*C))
    return y


def sys_dumpingLC_xi_yu_y(t, x, params):

    R = params['R']; L = params['L']; C = params['C']
    y = np.exp(-t*L/R)
    return np.real(y)