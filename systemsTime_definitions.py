# Import the system functions first
from systemsTime_equations import (
    sys_resistor_xi_yu_y,
    sys_resistor_xu_yi_y,
    sys_capacitor_xi_yu_y,
    sys_capacitor_xu_yi_y,
    sys_inductor_xi_yu_y,
    sys_inductor_xu_yi_y,
    sys_serialRC_xi_yu_y,
    sys_resistor_RT_xi_yu_y,
    system_1N4148,
    sys_nonlinearSq,
    sys_saturating,
    sys_CircuitRLC_xi_yu_y,
    sys_dumpingRC_xi_yu_y,
    sys_dumpingLC_xi_yu_y
)

def get_systems(tr):
    """
    Returns the systems dictionary using the provided translation function `tr`.
    Each class can pass its own self.tr.
    """
    return {
        tr("Resistor, x=i, y=u"): sys_resistor_xi_yu_y,
        tr("Resistor, x=u, y=i"): sys_resistor_xu_yi_y,
        tr("Capacitor, x=i, y=u"): sys_capacitor_xi_yu_y,
        tr("Capacitor, x=u, y=i"): sys_capacitor_xu_yi_y,
        tr("Inductor, x=i, y=u"): sys_inductor_xi_yu_y,
        tr("Inductor, x=u, y=i"): sys_inductor_xu_yi_y,
        tr("Serial RC, x=i, y=u"): sys_serialRC_xi_yu_y,
        tr("Resistor - R(T), x=i, y=u"): sys_resistor_RT_xi_yu_y,
        tr("Diode - nonlinear char."): system_1N4148,
        tr("Nonlinear: y = x^2"): sys_nonlinearSq,
        tr("Saturating: y = tanh(x)"): sys_saturating,
        tr("Circuit RLC, x=i, y=u"): sys_CircuitRLC_xi_yu_y,
        tr("Serial RC, x=i, y=u"): sys_dumpingRC_xi_yu_y,
        tr("Serial LC, x=i, y=u"): sys_dumpingLC_xi_yu_y
    }



