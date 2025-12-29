# Import the system functions first
from systemsH_equations import (
    sys_1_H_s,
    sys_2_H_s,
    sys_3_H_s,
    #sys_1_matZ_s,
    #sys_2_matZ_s
)

def get_systems(tr):
    """
    Returns the systems dictionary using the provided translation function `tr`.
    Each class can pass its own self.tr.
    """
    return {
        #tr("System 1"): sys_1_H,
        #tr("System 2"): sys_2_H,
        #tr("System 3"): sys_3_H,
        tr("System 1"): sys_1_H_s,
        tr("System 2"): sys_2_H_s,
        tr("System 3"): sys_3_H_s
        #tr("Two-port network 1"): sys_1_matZ_s,
        #tr("Two-port network 2"): sys_2_matZ_s
    }



