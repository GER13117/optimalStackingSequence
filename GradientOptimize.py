# helper function to find amount of halfwaves for minimal stability
# for finding the halfwaves to make the plates as unstable as possible it is only necessary to use sig_biax as tau_cr is idemepent of the max number of halfwaves
import MaterialFunc as mf
import numpy as np
from scipy.optimize import minimize

def optimizeAngles(numLayers, initialSymAngles, iterCount, maxDecimals, N_x, Tau, beta, b, alpha, tLayer, knockDown, maxHalfwaves):
    # Define bounds for angles
    m_angleBounds = [(-90, 90) for _ in range(numLayers)]

    # BALANCED PLY
    m_con1 = {'type': 'eq', 'fun': lambda sym_angles: mf.A_mat(sym_angles, tLayer)[0][2]}
    m_con2 = {'type': 'eq', 'fun': lambda sym_angles: mf.A_mat(sym_angles, tLayer)[1][2]}

    # PLY DOES NOT BUCKLE
    m_con3 = {'type': 'ineq',
              'fun': lambda sym_angles: 1 - mf.R_panelbuckling_comb(sym_angles, alpha, b, beta, tLayer, N_x, Tau, knockDown, maxHalfwaves)}

    # ANGLES ONLY HAVE SOME AMOUNT OF DECIMALS TODO: FIX NEEDED
    m_con4 = {'type': 'eq',
              'fun': lambda sym_angles: np.amax(abs(np.around(sym_angles, decimals=maxDecimals) - sym_angles))}

    # m_con6 = "10% of each ply share"
    m_cons = [m_con1, m_con2, m_con3]

    m_options = {'maxiter': iterCount}
    m_optimal_R = minimize(mf.R_panelbuckling_comb, initialSymAngles, method='trust-constr', bounds=m_angleBounds,
                           constraints=m_cons,
                           options=m_options,
                           args=(alpha, b, beta, tLayer, N_x, Tau, knockDown, maxHalfwaves))
    print(m_optimal_R)
    return m_optimal_R

def optimalLayers(initAngle: float, minLayers: int, maxLayers: int, maxDecimals: int, N_x: float, Tau: float,
                  beta: float, b: float, alpha: float, tLayer: float, knockDown: float, maxHalfwaves: int):
    for numLayers in range(int(minLayers / 2), int(maxLayers / 2) + 1):
        initialSymAngles = np.empty(numLayers)
        initialSymAngles.fill(initAngle)
        # initial optimization attempt, in order to find minimal amount of plies
        solution = optimizeAngles(numLayers, initialSymAngles, 1000, maxDecimals, N_x, Tau, beta, b, alpha, tLayer, knockDown, maxHalfwaves)
        print(numLayers, solution)
        if solution.fun < 1:  # TODO: Experiment with different values close to 1
            optimalSymLayers = numLayers
            optimalSymAngles = solution.x
            print(np.concatenate((optimalSymAngles, optimalSymAngles[::-1])))
            print(optimalSymLayers * 2)
            print(1 / solution.fun)

            # second optimization step in order to maximize strength of found minimal layer count
            print("Second Optimization Step")
            solution = optimizeAngles(optimalSymLayers, optimalSymAngles, 10000, maxDecimals, N_x, Tau, beta, b, alpha, tLayer, knockDown, maxHalfwaves)
            optimalSymLayers = numLayers
            optimalSymAngles = solution.x
            print("A:")
            print(mf.A_mat(optimalSymAngles, tLayer))
            print("B:")
            print(mf.B_mat(optimalSymAngles, tLayer))
            print("D:")
            print(mf.D_mat(optimalSymAngles, tLayer))
            print(np.concatenate((optimalSymAngles, optimalSymAngles[::-1])))
            print(optimalSymLayers * 2)
            print(1 / solution.fun)

            return solution
