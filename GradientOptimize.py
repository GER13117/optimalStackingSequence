# helper function to find amount of halfwaves for minimal stability
# for finding the halfwaves to make the plates as unstable as possible it is only necessary to use sig_biax as tau_cr is idemepent of the max number of halfwaves
import MaterialFunc as mf
import numpy as np
from scipy.optimize import minimize


def halfwavesObj(hWaves, b, t, beta, alpha, D11, D12, D22, D66):
    m, n = hWaves
    return mf.sig_x_cr_biax(b, t, beta, alpha, round(m), round(n), D11, D12, D22, D66)


def R_panelbuckling_comb(sym_angles, N_x, Tau, beta, b, alpha):
    print(3)
    m_numLayers = len(sym_angles) * 2
    m_hPanel = m_numLayers * mf.tLayer
    m_matD = mf.D_mat(sym_angles)

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

    print(4)
    # critical loads
    m_sigma_x_cr = m_mostUnstablePlate.fun
    m_tau_cr = mf.tau_cr(b, m_hPanel, m_D11, m_D12, m_D22, m_D66)

    # applied loads
    m_sigma_x = N_x / m_hPanel
    m_tau = Tau / m_hPanel

    # R values
    m_R_biax = abs(m_sigma_x / m_sigma_x_cr)
    m_R_shear = abs(m_tau / m_tau_cr)

    m_R = m_R_biax + m_R_shear ** 2

    # Debug - remove - hurts performance
    print("RF:", 1 / m_R, "- numLayers:", m_numLayers, "- numHalfwaves: ", m_mostUnstablePlate.x, "- Angles:",
          sym_angles)
    return m_R


def optimizeAngles(numLayers, initialSymAngles, iterCount, maxDecimals, N_x, Tau, beta, b, alpha):
    # Define bounds for angles
    m_angleBounds = [(-90, 90) for _ in range(numLayers)]

    # BALANCED PLY
    m_con1 = {'type': 'eq', 'fun': lambda sym_angles: mf.A_mat(sym_angles)[0][2]}
    m_con2 = {'type': 'eq', 'fun': lambda sym_angles: mf.A_mat(sym_angles)[1][2]}

    # PLY DOES NOT BUCKLE
    m_con3 = {'type': 'ineq',
              'fun': lambda sym_angles: 1 - R_panelbuckling_comb(sym_angles, N_x, Tau, beta, b, alpha)}

    # ANGLES ONLY HAVE SOME AMOUNT OF DECIMALS TODO: FIX NEEDED
    m_con4 = {'type': 'eq',
              'fun': lambda sym_angles: np.amax(abs(np.around(sym_angles, decimals=maxDecimals) - sym_angles))}

    # m_con5 = "strength check"  # TODO: Necessary?
    # m_con6 = "10% of each ply share"
    m_cons = [m_con1, m_con2, m_con3]

    m_options = {'maxiter': iterCount}
    print(2)
    m_optimal_R = minimize(R_panelbuckling_comb, initialSymAngles, method='trust-constr', bounds=m_angleBounds,
                           constraints=m_cons,
                           options=m_options, args=(N_x, Tau, beta, b, alpha))
    return m_optimal_R

def optimalLayers(initAngle: float, minLayers: int, maxLayers: int, maxDecimals: int, N_x: float, Tau: float,
                  beta: float, b: float, alpha: float):
    for numLayers in range(int(minLayers / 2), int(maxLayers / 2) + 1):
        initialSymAngles = np.empty(numLayers)
        initialSymAngles.fill(initAngle)
        print(1)
        # initial optimization attempt, in order to find minimal amount of plies
        solution = optimizeAngles(numLayers, initialSymAngles, 1000, maxDecimals, N_x, Tau, beta, b, alpha)
        print(numLayers, solution)
        if solution.fun < 1:  # TODO: Experiment with different values close to 1
            optimalSymLayers = numLayers
            optimalSymAngles = solution.x
            print(np.concatenate((optimalSymAngles, optimalSymAngles[::-1])))
            print(optimalSymLayers * 2)
            print(1 / solution.fun)

            # second optimization step in order to maximize strength of found minimal layer count
            print("Second Optimization Step")
            solution = optimizeAngles(optimalSymLayers, optimalSymAngles, 10000, maxDecimals, N_x, Tau, beta, b, alpha)
            optimalSymLayers = numLayers
            optimalSymAngles = solution.x
            print("A:")
            print(mf.A_mat(optimalSymAngles))
            print("B:")
            print(mf.B_mat(optimalSymAngles))
            print("D:")
            print(mf.D_mat(optimalSymAngles))
            print(np.concatenate((optimalSymAngles, optimalSymAngles[::-1])))
            print(optimalSymLayers * 2)
            print(1 / solution.fun)

            return solution
