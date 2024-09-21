import sys

import numpy as np

########################## PARAMETERS ##########################

from main import E11, E22, G12, tLayer, b, alpha, beta, N_x, Tau

########################## CALCULATIONS ##########################
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

def D_mat(angles):
    m_numLayers = len(angles)
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
        Q_bars[k] = Q_bar(angles[k])

    D_k = [0] * m_numLayers

    for k in range(m_numLayers):
        D_k[k] = Q_bars[k] * diff_zk3[k]
    D = np.array(D_k).sum(axis=0) / 3

    return D


def D_mat_sym(sym_angles):
    m_angles = np.concatenate((sym_angles, sym_angles[::-1]))
    return D_mat(m_angles)


def B_mat(angles):
    m_numLayers = len(angles)
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
        Q_bars[k] = Q_bar(angles[k])

    B_k = [0] * m_numLayers

    for k in range(m_numLayers):
        B_k[k] = Q_bars[k] * diff_zk2[k]
    B = np.array(B_k).sum(axis=0) / 3

    return B


def B_mat_sym(sym_angles):
    m_angles = np.concatenate((sym_angles, sym_angles[::-1]))
    return B_mat(m_angles)


def A_mat(angles):
    m_numLayers = len(angles)
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
        Q_bars[k] = Q_bar(angles[k])

    A_k = [0] * m_numLayers

    for k in range(m_numLayers):
        A_k[k] = Q_bars[k] * diff_zk[k]
    A = np.array(A_k).sum(axis=0)

    return A


def A_mat_sym(sym_angles):
    m_angles = np.concatenate((sym_angles, sym_angles[::-1]))
    return A_mat(m_angles)


# biaxial
def sig_x_cr_biax(hPanel: float, m: float, n: float, D11: float, D12: float, D22: float,
                  D66: float):
    return np.pi ** 2 / b ** 2 / hPanel / ((m / alpha) ** 2 + beta * n ** 2) * (
            D11 * (m / alpha) ** 4 + 2 * (D12 + D66) * (m * n / alpha) ** 2 + D22 * n ** 4)


# shear
def tau_cr(hPanel: float, D11: float, D12: float, D22: float, D66: float):
    delta = np.sqrt(D11 * D22) / (D12 + 2 * D66)  # stiffeness ratio
    if delta >= 1:
        return 4 / hPanel / b ** 2 * ((D11 * D22 ** 3) ** (1 / 4) * (8.12 + 5.05 / delta))
    else:
        return 4 / hPanel / b ** 2 * (np.sqrt(D22 * (D12 + 2 * D66)) * (11.7 + 0.532 * delta + 0.938 * delta ** 2))


def R_panelbuckling_comb(stackingSeq: list, knockDown: float, maxHalfwaves: int):
    m_matD = D_mat(stackingSeq)
    m_D11 = m_matD[0][0] * knockDown
    m_D12 = m_matD[0][1] * knockDown
    m_D22 = m_matD[1][1] * knockDown
    m_D66 = m_matD[2][2] * knockDown

    m_hPanel = tLayer * len(stackingSeq)

    m_sigma_x_cr = sys.float_info.max
    for m in range(1, maxHalfwaves + 1):
        for n in range(1, maxHalfwaves + 1):
            m_sig_x_xr_new = sig_x_cr_biax(m_hPanel, m, n, m_D11, m_D12, m_D22, m_D66)
            if 0 < m_sig_x_xr_new < m_sigma_x_cr:
                m_sigma_x_cr = m_sig_x_xr_new

    m_tau_cr = tau_cr(m_hPanel, m_D11, m_D12, m_D22, m_D66)
    # applied loads
    m_sigma_x = N_x / m_hPanel
    m_tau = Tau / m_hPanel

    # R values
    m_R_biax = abs(m_sigma_x / m_sigma_x_cr) * 1.5  # 1.5 for ultimate loading
    m_R_shear = abs(m_tau / m_tau_cr) * 1.5

    m_R = m_R_biax + m_R_shear ** 2

    return m_R


def R_panelbuckling_comb_sym(stackingSeqSym: list, knockDown: float, maxHalfwaves: int):
    m_angles = np.concatenate((stackingSeqSym, stackingSeqSym[::-1])).tolist()
    return R_panelbuckling_comb(m_angles, knockDown, maxHalfwaves)