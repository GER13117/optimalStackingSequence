from scipy.optimize import minimize
import numpy as np

# Einschränkungen: nur symmetrische Laminate mit gerader Anzahl von schichten TODO: Check if statement is true
a = 400  # 400mm-800mm
b = 250  # 150mm-250mm
alpha = a / b
N_x = -1000  # -500N to -2000N
N_y = -500  # 300N to -1000N
Tau = 100  # 100 to 800N

beta = N_y / N_x
tLayer = 0.184  # 0.184mm to 0.25mm

E11 = 130_000  # 130_000 to 160_000 MPa
E22 = 10_000  # 10_000 to 14_000 MPa
G12 = 5_000  # 5_000 to 8_000 MPa

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


def sig_x_cr(b, t, beta, alpha, m, n, D11, D12, D22, D66):
    return np.pi ** 2 / b ** 2 / t / ((m / alpha) ** 2 + beta * n ** 2) * (
            D11 * (m / alpha) ** 4 + 2 * (D12 + D66) * (m * n / alpha) ** 2 + D22 * n ** 4)


# helper function to find amount of halfwaves for minimal stability
def halfwavesObj(hWaves, b, t, beta, alpha, D11, D12, D22, D66):
    m, n = hWaves
    return sig_x_cr(b, t, beta, alpha, round(m), round(n), D11, D12, D22, D66)


def R_panelbuckling(sym_angles):
    m_numLayers = len(sym_angles) * 2
    m_hPanel = m_numLayers * tLayer
    matD = D_mat(sym_angles)

    D11 = matD[0][0]
    D12 = matD[0][1]
    D22 = matD[1][1]
    D66 = matD[2][2]

    initialHWaves = np.ones(2)
    hWaveBnd = (1, 5)
    hWavesBnds = [hWaveBnd, hWaveBnd]

    result = minimize(halfwavesObj, initialHWaves, method='SLSQP', bounds=hWavesBnds,
                      args=(b, m_hPanel, beta, alpha, D11, D12, D22, D66))

    sigma_x_cr = result.fun
    sigma_x = N_x / m_hPanel
    R = abs(sigma_x / sigma_x_cr)
    print(1 / R, ", ", m_numLayers)
    return R


def optimizeAngles(numLayers, initialSymAngles, iterCount):
    # Define bounds for angles
    angleBounds = [(-90, 90) for _ in range(numLayers)]

    con1 = {'type': 'eq', 'fun': lambda sym_angles: A_mat(sym_angles)[0][2]}  # balanced
    con2 = {'type': 'eq', 'fun': lambda sym_angles: A_mat(sym_angles)[1][2]}  # balanced
    con3 = {'type': 'ineq', 'fun': lambda sym_angles: 1 - R_panelbuckling(sym_angles)}  # panel does not buckle
    con4 = "strength check"
    con5 = "10% of each ply share"
    cons = [con1, con2, con3]

    options = {'maxiter': iterCount}

    solution = minimize(R_panelbuckling, initialSymAngles, method='trust-constr', bounds=angleBounds, constraints=cons,
                        options=options)
    return solution


def T_mat(theta):
    theta_rad = np.deg2rad(theta)
    return np.array([
        [np.cos(theta_rad) ** 2, np.sin(theta_rad) ** 2, 2 * np.sin(theta_rad) * np.cos(theta_rad)],
        [np.sin(theta_rad) ** 2, np.cos(theta_rad) ** 2, -2 * np.sin(theta_rad) * np.cos(theta_rad)],
        [-np.sin(theta_rad) * np.cos(theta_rad), np.sin(theta_rad) * np.cos(theta_rad),
         np.cos(theta_rad) ** 2 - np.sin(theta_rad) ** 2]
    ])


def RF_strength(sym_angles):
    numLayers = len(sym_angles) * 2
    h = numLayers * tLayer
    # Inverse of ABD
    # Multiply applied Loads with ABD^-1 => epsylon_0 and k
    # calculate epsylon_x, epsylon_y, gamma_xy for each ply
    # transform all epsylon into material cos
    # use Q matrix to calculate each sigma_1, sigma_2, sigma_12
    # calculate RF_ff and RF_iff for each layer
    # return appropriate value for failure criterion


minLayers = 22
maxLayers = 22
optimalLayers = maxLayers
optimalAngles = []


def optimalLayers(initAngle):
    # check multiple layers with low iteration count
    # if threshold (maybe R<1 or very close to 1) is reached after n amount of iterations run optimization for this layer count with high amount of iterations, use old final values as initial values.
    for numLayers in range(minLayers, maxLayers + 1):
        initialSymAngles = np.empty(numLayers)
        initialSymAngles.fill(initAngle)
        solution = optimizeAngles(numLayers, initialSymAngles, 1000)
        print(numLayers, solution)
        if solution.fun < 1:
            optimalLayers = numLayers
            optimalAngles = solution.x
            print("D:")
            print(D_mat(optimalAngles))
            print("A:")
            print(A_mat(optimalAngles))
            print(np.concatenate((optimalAngles, optimalAngles[::-1])))
            print(optimalLayers * 2)
            print(1 / solution.fun)
            print("Second Optimization Step")
            solution = optimizeAngles(optimalLayers, optimalAngles, 10000)
            print(1 / solution.fun)
            optimalLayers = numLayers
            optimalAngles = solution.x
            return solution


optimalLayers(45)
# TODO: Add functionality to avoid local minima
