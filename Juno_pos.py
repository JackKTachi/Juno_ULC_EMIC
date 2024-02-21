# Juno_pos.py

import numpy as np
import math
import spiceypy as spice

# temp

# Juno's position relative to the magnetic field represented by the VIP4 field model [km].
spice.furnsh('kernel/cassMetaK.txt')
jun_et = spice.str2et('2016 189 // 0:0:0.648')
pos, lightTimes = spice.spkpos(
    targ='JUNO', et=jun_et, ref='IAU_JUPITER', abcorr='none', obs='JUPITER'
)
# print(pos/RJ)
rho = np.sqrt(pos[0]**2 + pos[1]**2)
Z = pos[2]


def S3(pos_arr):
    """
    Args:
        pos_arr (1d-ndarray): object position in IAU_JUPITER [m]

    Returns:
        Sys3: System III longitude of the object [deg]
        posphi: longitude in S3RH [deg]
    """

    posx, posy, posz = pos_arr[0], pos_arr[1], pos_arr[2]
    # posr = np.sqrt(posx**2 + posy**2 + posz**2)
    # postheta = np.arccos(posz/posr)
    posphi = np.arctan2(posy, posx)
    if posphi < 0:
        Sys3 = np.degrees(-posphi)
    else:
        Sys3 = np.degrees(2*np.pi - posphi)

    return Sys3, np.degrees(posphi)


def lt(utc):
    """_summary_

    Args:
        utc: observation date

    Returns:
        elong: east longitude of Europa (0 is defined at the anti-solar position) [deg]
        td: local time [sec]
    """

    # The sun's position seen from the Jupiter in IAU_JUPITER coordinate.
    posSUN, _ = spice.spkpos(
        targ='SUN', et=utc, ref='IAU_JUPITER', abcorr='LT+S', obs='JUPITER'
    )

    # Europa's position seen from the Jupiter in IAU_JUPITER coordinate.
    posEUR, _ = spice.spkpos(
        targ='EUROPA', et=utc, ref='IAU_JUPITER', abcorr='NONE', obs='JUPITER'
    )

    dot = posSUN[0]*1+posSUN[1]*0
    R_SUN = math.sqrt(posSUN[0]**2+posSUN[1]**2+posSUN[2]**2)
    arg = math.degrees(math.acos(dot/(R_SUN)))

    sec = (3600*24/360)*arg    # [sec]
    # print(elong)
    # print(td)
    return arg, sec
