from scipy.optimize import minimize
import numpy as np

# Einschränkungen: nur symmetrische Laminate mit gerader Anzahl von schichten
a = 600
b = 400
alpha = a / b
sigma_x = -43.051
sigma_y = 10.122
beta = sigma_y / sigma_x
tLayer = 0.184

E11 = 132741.56
E22 = 10210.89
G12 = 5105.44
nu12 = 0.33
nu21 = E22 / E11 * nu12

Q11 = E11 / (1 - nu12 * nu21)
Q12 = nu12 * E22 / (1 - nu12 * nu21)
Q22 = E22 / (1 - nu21 * nu12)
Q66 = G12

Q = np.array([[Q11, Q12, 0],
              [Q12, Q22, 0],
              [0, 0, Q66]
              ])


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


def D_mat(sym_angles):
    m_numLayers = len(sym_angles) * 2
    m_angles = np.concatenate((sym_angles, sym_angles[::-1]))
    m_hPanel = m_numLayers * tLayer

    # array of all z_k
    m_zks = np.zeros(m_numLayers + 1)
    m_zks[0] = -m_hPanel / 2
    for zk in range(1, m_numLayers + 1):
        m_zks[zk] = m_zks[zk - 1] + tLayer

    # array of z_k+1^3-z_k^3
    diff_zk3s = np.zeros(m_numLayers)
    for k in range(m_numLayers):
        diff_zk3s[k] = m_zks[k + 1] ** 3 - m_zks[k] ** 3

    Q_bars = [0] * m_numLayers
    for k in range(m_numLayers):
        Q_bars[k] = Q_bar(m_angles[k])

    D_ks = [0] * m_numLayers

    for k in range(m_numLayers):
        D_ks[k] = Q_bars[k] * diff_zk3s[k]
    D = np.array(D_ks).sum(axis=0) / 3

    return D


def A_mat(sym_angles):
    m_numLayers = len(sym_angles) * 2
    m_angles = np.concatenate((sym_angles, sym_angles[::-1]))
    m_hPanel = m_numLayers * tLayer

    # array of all z_k
    m_zks = np.zeros(m_numLayers + 1)
    m_zks[0] = -m_hPanel / 2
    for zk in range(1, m_numLayers + 1):
        m_zks[zk] = m_zks[zk - 1] + tLayer

    # array of z_k+1^3-z_k^3
    diff_zks = np.zeros(m_numLayers)
    for k in range(m_numLayers):
        diff_zks[k] = m_zks[k + 1] - m_zks[k]

    Q_bars = [0] * m_numLayers
    for k in range(m_numLayers):
        Q_bars[k] = Q_bar(m_angles[k])

    A_ks = [0] * m_numLayers

    for k in range(m_numLayers):
        A_ks[k] = Q_bars[k] * diff_zks[k]
    A = np.array(A_ks).sum(axis=0)

    return A


def sig_x_cr(b, t, beta, alpha, m, n, D11, D12, D22, D66):
    return np.pi ** 2 / b ** 2 / t / ((m / alpha) ** 2 + beta * n ** 2) * (
            D11 * (m / alpha) ** 4 + 2 * (D12 + D66) * (m * n / alpha) ** 2 + D22 * n ** 4)


def halfwavesObj(hWaves, b, t, beta, alpha, D11, D12, D22, D66):
    m, n = hWaves
    return sig_x_cr(b, t, beta, alpha, m, n, D11, D12, D22, D66)


def R_panelbuckling(sym_angles):
    numLayers = len(sym_angles) * 2
    t = numLayers * tLayer
    matD = D_mat(sym_angles)

    D11 = matD[0][0]
    D12 = matD[0][1]
    D22 = matD[1][1]
    D66 = matD[2][2]

    initialHWaves = np.ones(2)
    hWaveBnd = (1, 5)
    hWavesBnds = [hWaveBnd, hWaveBnd]

    result = minimize(halfwavesObj, initialHWaves, method='SLSQP', bounds=hWavesBnds,
                      args=(b, t, beta, alpha, D11, D12, D22, D66))

    m, n = result.x
    sigma_x_cr = sig_x_cr(b, t, beta, alpha, round(m), round(n), D11, D12, D22, D66)
    return abs(sigma_x / sigma_x_cr)


def optimizeAngles(numLayers, initAngle):
    initialAngles = np.empty(numLayers)
    initialAngles.fill(initAngle)

    # Define bounds for angles
    angleBounds = [(-90, 90) for _ in range(numLayers)]

    con1 = {'type': 'eq', 'fun': lambda sym_angles: A_mat(sym_angles)[0][2]}  # balanced
    con2 = {'type': 'eq', 'fun': lambda sym_angles: A_mat(sym_angles)[1][2]}  # balanced
    cons = [con1, con2]

    solution = minimize(R_panelbuckling, initialAngles, method='SLSQP', bounds=angleBounds, constraints=cons)
    return solution


minLayers = 18
maxLayers = 40
optimalLayers = maxLayers
optimalAngles = []

def optimalLayers(initAngle):
    for numLayers in range(minLayers, maxLayers + 1):
        solution = optimizeAngles(numLayers, initAngle)
        print(numLayers, solution)
        if solution.success and solution.fun < 1:
            optimalLayers = numLayers
            optimalAngles = solution.x
            print("D:")
            print(D_mat(optimalAngles))
            print("A:")
            print(A_mat(optimalAngles))
            print(np.concatenate((optimalAngles, optimalAngles[::-1])))
            print(optimalLayers * 2)
            print(solution.fun)
            return solution

optimalLayers(45)
# TODO: Add functionality to avoid local minima