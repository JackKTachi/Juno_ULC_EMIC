# Juno_pos.py

import numpy as np
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
