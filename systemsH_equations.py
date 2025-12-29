import numpy as np

'''
def sys_RR2LC_H(s, params):
    R, R2, L, C = params['R'], params['R2'], params['L'], params['C']

    det = (L*s+R)*(L*s+R2+1/(C*s)) - (L*s)**2 if np.abs(s*C) > 0 else np.inf

    H = L / (C*det) if (C>0) else 0
    return H



def sys_1_H(s, params):
    R, R2, L, C = params['R'], params['R2'], params['L'], params['C']
    Y = s*L / (R + s*L)
    return Y

def sys_2_H(s, params):
    R, R2, L, C = params['R'], params['R2'], params['L'], params['C']
    return (R2*C*L*s*s) / (R + L*s + R*R2*C*s + R*L*C*s*s + R2*L*C*s*s)
    #return R / (R + (R+s*L)*(1 + s*R*C))

def sys_3_H(s, params):
    R, R2, L, C = params['R'], params['R2'], params['L'], params['C']
    if s*s*L*C != -1:
        H = (s*L) / (1 + s*s*L*C)
    else:
        H = np.inf
    return H

'''

# --- Example system transfer functions H(jw) ---
#def sys_RR2LC_H_s(s, params):
#    R, R2, L, C = params['R'], params['R2'], params['L'], params['C']

#    det = (L*s+R)*(L*s+R2+1/(C*s)) - (L*s)**2 if np.abs(s*C) > 0 else np.inf
#    H = L / (C*det) if (C>0) else 0
#    return H

def sys_1_H_s(s, params):
    R, L, C = params['R'], params['L'], params['C']

    Y = s*L / (R + s*L)
    return Y

def sys_2_H_s(s, params):
    R, R2, L, C = params['R'], params['R2'], params['L'], params['C']

    return (R2*C*L*s*s) / (R + L*s + R*R2*C*s + R*L*C*s*s + R2*L*C*s*s)
    #return R / (R + (R+s*L)*(1 + s*R*C))

def sys_3_H_s(s, params):
    R, L, C = params['R'], params['L'], params['C']
    
    H = (s*C) / (1 + s*R*C + s*s*L*C)
    #if s*s*L*C != -1:
    #    H = (s*L) / (1 + s*s*L*C)
    #else:
    #    H = np.inf
    return H


'''
# ---- Z-Matrix functions ----
def sys_1_matZ_s(s, R, L, C, Zb):
    Z11 = s*L + (1/(s*C) if np.abs(s*C) != 0 else 1e12)
    Z22 = R + (1/(s*C) if np.abs(s*C) != 0 else 1e12)
    Z12 = Z21 = (-1/(s*C) if np.abs(s*C) != 0 else 1e12)
    return np.array([[Z11, Z12], [Z21, Z22]], dtype=complex)

def sys_2_matZ_s(s, R, L, C, Zb):
    Z11 = R + 2*s*L
    Z22 = R + Zb + (1/(s*C) if np.abs(s*C) != 0 else 1e12)
    Z12 = Z21 = (s*L) / 4
    return np.array([[Z11, Z12], [Z21, Z22]], dtype=complex)
'''

