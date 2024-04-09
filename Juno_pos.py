# Juno_pos.py

import numpy as np
import math
import spiceypy as spice
import datetime

# Constants
RJ = 71492000   # Jupiter's equatorial radius [m]


def S3(pos_arr):
    """
    Args:
        pos_arr (1d-ndarray): object position in IAU_JUPITER (planet centric) [m]

    Returns:
        Sys3: System III longitude of the object [deg]
    """

    posx, posy, posz = pos_arr[0], pos_arr[1], pos_arr[2]
    # posr = np.sqrt(posx**2 + posy**2 + posz**2)
    # postheta = np.arccos(posz/posr)
    posphi = np.arctan2(posy, posx)
    if posphi < 0:
        Sys3 = np.degrees(-posphi)
    else:
        Sys3 = np.degrees(2*np.pi - posphi)

    return Sys3


def Clat(pos_arr):
    """
    Args:
        pos_arr (1d-ndarray): object position in IAU_JUPITER (planet centric) [m]

    Returns:
        Clat: centrifugal latitude of the object [deg] (0 = centrifugal equator)
    """
    posx, posy, posz = pos_arr[0], pos_arr[1], pos_arr[2]
    R = math.sqrt(posx**2 + posy**2 + posz**2)
    theta = np.arccos(posz/R)
    phi = np.arctan2(posy, posx)

    # https://doi.org/10.1029/2020JA028713
    a, b, c, d, e = 1.66, 0.131, 1.62, 7.76, 249
    CL = (a*math.tanh(b*R/RJ-c)+d) * \
        math.sin(math.radians(math.degrees(phi)-159.2-e))
    Clat = 90-math.degrees(theta)-CL

    return Clat


def lt_iauJup(self, TARGET: str, utc):
    """
    Args:
        TARGET: target in the Jupiter system (like EUROPA and JUNO)
        utc: observation date at the target (UT)

    Returns:
        td: local time in datetime [hh:mm:ss]
    """
    et = spice.str2et(utc)

    # Eigen vector toward the sun's position seen from Jupiter in IAU_JUPITER coordinate.
    posSUN, lighttime = spice.spkpos(
        targ='SUN', et=et, ref='IAU_JUPITER', abcorr='LT+S', obs='JUPITER'
    )
    posSUN = posSUN/math.sqrt(posSUN[0]**2 + posSUN[1]**2 + posSUN[2]**2)

    # Eigen vector toward target's position seen from Jupiter in IAU_JUPITER coordinate.
    posTARG, _ = spice.spkpos(
        targ=TARGET, et=et, ref='IAU_JUPITER', abcorr='NONE', obs='JUPITER'
    )
    posTARG = posTARG / \
        math.sqrt(posTARG[0]**2 + posTARG[1]**2 + posTARG[2]**2)

    # Dusk terminator
    dusk_term = np.array([-posSUN[1], posSUN[0]])
    dusk_dot = dusk_term[0]*posTARG[0] + dusk_term[1]*posTARG[1]
    if dusk_dot >= 0:
        # Target is in the dusk side.
        d_phi = np.pi + \
            math.acos(posSUN[0]*posTARG[0]+posSUN[1]*posTARG[1])
        print('Dusk [deg]:', math.degrees(d_phi))
    else:
        # Target is in the dawn side.
        d_phi = np.pi - \
            math.acos(posSUN[0]*posTARG[0] + posSUN[1]*posTARG[1])
        print('Dawn [deg]:', math.degrees(d_phi))

    sec = (3600*24/360)*math.degrees(d_phi)    # [sec]
    td = datetime.timedelta(seconds=sec)
    print(td)

    return td
