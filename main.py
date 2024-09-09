from scipy.optimize import minimize
import numpy as np

# Einschränkungen: nur symmetrische Laminate mit gerader Anzahl von schichten TODO: Check if statement is true
# TODO: Knockdown Factor
# TODO: Constraints - 10% plyshare, strength?, only integer values for angles
########################## PARAMETERS ##########################
a = 400  # 400mm-800mm
b = 250  # 150mm-250mm
alpha = a / b
N_x = -1000  # -500N to -2000N
N_y = -500  # 300N to -1000N
Tau = 100  # 100 to 800N

tLayer = 0.184  # 0.184mm to 0.25mm

E11 = 130_000  # 130_000 to 160_000 MPa
E22 = 10_000  # 10_000 to 14_000 MPa
G12 = 5_000  # 5_000 to 8_000 MPa

minLayers = 40
maxLayers = 100
maxDecimals = 0 #0 => Angles of result are only integer values

########################## CALCULATIONS ##########################
beta = N_y / N_x
nu12 = 0.33  # TODO: Needs to be calculated????
nu21 = E22 / E11 * nu12

Q11 = E11 / (1 - nu12 * nu21)
Q12 = nu12 * E22 / (1 - nu12 * nu21)
Q22 = E22 / (1 - nu21 * nu12)
Q66 = G12


# Creates Q_bar matrix
def Q_bar(theta):
    theta_rad = np.deg2rad(theta)

    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    cos2_theta = cos_theta * cos_theta
    sin2_theta = sin_theta * sin_theta
    cos4_theta = cos2_theta * cos2_theta
    sin4_theta = sin2_theta * sin2_theta
    sin2_theta_cos2_theta = sin2_theta * cos2_theta

    Q11_bar = Q11 * cos4_theta + 2 * (Q12 + 2 * Q66) * sin2_theta_cos2_theta + Q22 * sin4_theta
    Q12_bar = (Q11 + Q22 - 4 * Q66) * sin2_theta_cos2_theta + Q12 * (sin4_theta + cos4_theta)
    Q22_bar = Q11 * sin4_theta + 2 * (Q12 + 2 * Q66) * sin2_theta_cos2_theta + Q22 * cos4_theta
    Q16_bar = (Q11 - Q12 - 2 * Q66) * sin_theta * cos_theta ** 3 + (Q12 - Q22 + 2 * Q66) * sin_theta ** 3 * cos_theta
    Q26_bar = (Q11 - Q12 - 2 * Q66) * cos_theta * sin_theta ** 3 + (Q12 - Q22 + 2 * Q66) * cos_theta ** 3 * sin_theta
    Q66_bar = (Q11 + Q22 - 2 * Q12 - 2 * Q66) * sin2_theta_cos2_theta + Q66 * (sin4_theta + cos4_theta)

    Q_bar = np.array([
        [Q11_bar, Q12_bar, Q16_bar],
        [Q12_bar, Q22_bar, Q26_bar],
        [Q16_bar, Q26_bar, Q66_bar]
    ])
    return Q_bar


# ABD Functions
def D_mat(sym_angles):
    m_numLayers = len(sym_angles) * 2
    m_angles = np.concatenate((sym_angles, sym_angles[::-1]))
    m_hPanel = m_numLayers * tLayer

    # array of all z_k
    m_zk = np.zeros(m_numLayers + 1)
    m_zk[0] = -m_hPanel / 2
    for zk in range(1, m_numLayers + 1):
        m_zk[zk] = m_zk[zk - 1] + tLayer

    # array of z_k+1^3-z_k^3
    diff_zk3 = np.zeros(m_numLayers)
    for k in range(m_numLayers):
        diff_zk3[k] = m_zk[k + 1] ** 3 - m_zk[k] ** 3

    Q_bars = [0] * m_numLayers
    for k in range(m_numLayers):
        Q_bars[k] = Q_bar(m_angles[k])

    D_k = [0] * m_numLayers

    for k in range(m_numLayers):
        D_k[k] = Q_bars[k] * diff_zk3[k]
    D = np.array(D_k).sum(axis=0) / 3

    return D


def B_mat(sym_angles):
    m_numLayers = len(sym_angles) * 2
    m_angles = np.concatenate((sym_angles, sym_angles[::-1]))
    m_hPanel = m_numLayers * tLayer

    # array of all z_k
    m_zk = np.zeros(m_numLayers + 1)
    m_zk[0] = -m_hPanel / 2
    for zk in range(1, m_numLayers + 1):
        m_zk[zk] = m_zk[zk - 1] + tLayer

    # array of z_k+1^3-z_k^3
    diff_zk2 = np.zeros(m_numLayers)
    for k in range(m_numLayers):
        diff_zk2[k] = m_zk[k + 1] ** 2 - m_zk[k] ** 2

    Q_bars = [0] * m_numLayers
    for k in range(m_numLayers):
        Q_bars[k] = Q_bar(m_angles[k])

    B_k = [0] * m_numLayers

    for k in range(m_numLayers):
        B_k[k] = Q_bars[k] * diff_zk2[k]
    B = np.array(B_k).sum(axis=0) / 3

    return B


def A_mat(sym_angles):
    m_numLayers = len(sym_angles) * 2
    m_angles = np.concatenate((sym_angles, sym_angles[::-1]))
    m_hPanel = m_numLayers * tLayer

    # array of all z_k
    m_zk = np.zeros(m_numLayers + 1)
    m_zk[0] = -m_hPanel / 2
    for zk in range(1, m_numLayers + 1):
        m_zk[zk] = m_zk[zk - 1] + tLayer

    # array of z_k+1-z_k
    diff_zk = np.zeros(m_numLayers)
    for k in range(m_numLayers):
        diff_zk[k] = m_zk[k + 1] - m_zk[k]

    Q_bars = [0] * m_numLayers
    for k in range(m_numLayers):
        Q_bars[k] = Q_bar(m_angles[k])

    A_k = [0] * m_numLayers

    for k in range(m_numLayers):
        A_k[k] = Q_bars[k] * diff_zk[k]
    A = np.array(A_k).sum(axis=0)

    return A


# biaxial
def sig_x_cr_biax(b, t, beta, alpha, m, n, D11, D12, D22, D66):
    return np.pi ** 2 / b ** 2 / t / ((m / alpha) ** 2 + beta * n ** 2) * (
            D11 * (m / alpha) ** 4 + 2 * (D12 + D66) * (m * n / alpha) ** 2 + D22 * n ** 4)


# shear
def tau_cr(b, t, D11, D12, D22, D66):
    delta = np.sqrt(D11 * D22) / (D12 + 2 * D66)  # stiffeness ratio
    if delta >= 1:
        return 4 / t / b ** 2 * ((D11 * D22 ** 3) ** (1 / 4) * (8.12 + 5.05 / delta))
    else:
        return 4 / t / b ** 2 * (np.sqrt(D22 * (D12 + 2 * D66)) * (11.7 + 0.532 * delta + 0.938 * delta ** 2))


# helper function to find amount of halfwaves for minimal stability
# for finding the halfwaves to make the plates as unstable as possible it is only necessary to use sig_biax as tau_cr is idemepent of the max number of halfwaves
def halfwavesObj(hWaves, b, t, beta, alpha, D11, D12, D22, D66):
    m, n = hWaves
    return sig_x_cr_biax(b, t, beta, alpha, round(m), round(n), D11, D12, D22, D66)


def R_panelbuckling_comb(sym_angles):
    m_numLayers = len(sym_angles) * 2
    m_hPanel = m_numLayers * tLayer
    m_matD = D_mat(sym_angles)

    m_D11 = m_matD[0][0]
    m_D12 = m_matD[0][1]
    m_D22 = m_matD[1][1]
    m_D66 = m_matD[2][2]

    m_initialHWaves = np.ones(2)  # [1,1]
    m_hWaveBnd = (1, 5)
    m_hWavesBnds = [m_hWaveBnd, m_hWaveBnd]

    # Find amount of halfwaves that make plate as unstable as possible
    m_mostUnstablePlate = minimize(halfwavesObj, m_initialHWaves, method='SLSQP', bounds=m_hWavesBnds,
                                   args=(b, m_hPanel, beta, alpha, m_D11, m_D12, m_D22, m_D66))

    # critical loads
    m_sigma_x_cr = m_mostUnstablePlate.fun
    m_tau_cr = tau_cr(b, m_hPanel, m_D11, m_D12, m_D22, m_D66)

    # applied loads
    m_sigma_x = N_x / m_hPanel
    m_tau = Tau / m_hPanel

    # R values
    m_R_biax = abs(m_sigma_x / m_sigma_x_cr)
    m_R_shear = abs(m_tau / m_tau_cr)

    m_R = m_R_biax + m_R_shear ** 2

    # Debug - remove - hurts performance
    print("RF:", 1 / m_R, "- numLayers:", m_numLayers, "- numHalfwaves: ", m_mostUnstablePlate.x, "- Angles:", sym_angles)
    return m_R


def optimizeAngles(numLayers, initialSymAngles, iterCount):
    # Define bounds for angles
    m_angleBounds = [(-90, 90) for _ in range(numLayers)]

    # BALANCED PLY
    m_con1 = {'type': 'eq', 'fun': lambda sym_angles: A_mat(sym_angles)[0][2]}
    m_con2 = {'type': 'eq', 'fun': lambda sym_angles: A_mat(sym_angles)[1][2]}

    # PLY DOES NOT BUCKLE
    m_con3 = {'type': 'ineq', 'fun': lambda sym_angles: 1 - R_panelbuckling_comb(sym_angles)}

    # ANGLES ONLY HAVE SOME AMOUNT OF DECIMALS TODO: FIX NEEDED
    m_con4 = {'type': 'eq', 'fun': lambda sym_angles: np.amax(abs(np.around(sym_angles, decimals=maxDecimals) - sym_angles))}

    # m_con5 = "strength check"  # TODO: Necessary?
    # m_con6 = "10% of each ply share"
    m_cons = [m_con1, m_con2, m_con3]

    m_options = {'maxiter': iterCount}

    m_optimal_R = minimize(R_panelbuckling_comb, initialSymAngles, method='trust-constr', bounds=m_angleBounds,
                           constraints=m_cons,
                           options=m_options)
    return m_optimal_R


def T_mat(theta):
    theta_rad = np.deg2rad(theta)
    return np.array([
        [np.cos(theta_rad) ** 2, np.sin(theta_rad) ** 2, 2 * np.sin(theta_rad) * np.cos(theta_rad)],
        [np.sin(theta_rad) ** 2, np.cos(theta_rad) ** 2, -2 * np.sin(theta_rad) * np.cos(theta_rad)],
        [-np.sin(theta_rad) * np.cos(theta_rad), np.sin(theta_rad) * np.cos(theta_rad),
         np.cos(theta_rad) ** 2 - np.sin(theta_rad) ** 2]
    ])


minSymLayers = int(minLayers / 2)
maxSymLayers = int(maxLayers / 2)


def optimalLayers(initAngle):
    for numLayers in range(minSymLayers, maxSymLayers + 1):
        initialSymAngles = np.empty(numLayers)
        initialSymAngles.fill(initAngle)

        # initial optimization attempt, in order to find minimal amount of plies
        solution = optimizeAngles(numLayers, initialSymAngles, 1000)
        print(numLayers, solution)
        if solution.fun < 1:  # TODO: Experiment with different values close to 1
            optimalSymLayers = numLayers
            optimalSymAngles = solution.x
            print(np.concatenate((optimalSymAngles, optimalSymAngles[::-1])))
            print(optimalSymLayers * 2)
            print(1 / solution.fun)

            # second optimization step in order to maximize strength of found minimal layer count
            print("Second Optimization Step")
            solution = optimizeAngles(optimalSymLayers, optimalSymAngles, 10000)
            optimalSymLayers = numLayers
            optimalSymAngles = solution.x
            print("A:")
            print(A_mat(optimalSymAngles))
            print("B:")
            print(B_mat(optimalSymAngles))
            print("D:")
            print(D_mat(optimalSymAngles))
            print(np.concatenate((optimalSymAngles, optimalSymAngles[::-1])))
            print(optimalSymLayers * 2)
            print(1 / solution.fun)

            return solution


optimalLayers(45)
