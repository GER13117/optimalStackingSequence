import MaterialFunc as mf
import numpy as np
from scipy.optimize import minimize

def optimizeAngles(numLayers, initialSymAngles, iterCount, knockDown, maxHalfwaves):
    # Define bounds for angles
    m_angleBounds = [(-90, 90) for _ in range(numLayers)]

    # BALANCED PLY
    m_con1 = {'type': 'eq', 'fun': lambda sym_angles: mf.A_mat_sym(sym_angles)[0][2]}
    m_con2 = {'type': 'eq', 'fun': lambda sym_angles: mf.A_mat_sym(sym_angles)[1][2]}

    # PLY DOES NOT BUCKLE
    m_con3 = {'type': 'ineq',
              'fun': lambda sym_angles: 1 - mf.R_panelbuckling_comb_sym(sym_angles, knockDown, maxHalfwaves)}

    m_cons = [m_con1, m_con2, m_con3]

    m_options = {'maxiter': iterCount}
    m_optimal_R = minimize(mf.R_panelbuckling_comb_sym, initialSymAngles, method='trust-constr', bounds=m_angleBounds,
                           constraints=m_cons,
                           options=m_options,
                           args=(knockDown, maxHalfwaves))
    return m_optimal_R

def optimalLayers(initAngle: float, minLayers: int, maxLayers: int, knockDown: float, maxHalfwaves: int, minRF=1.0):
    for numLayers in range(int(minLayers / 2), int(maxLayers / 2) + 1):
        initialSymAngles = np.empty(numLayers)
        initialSymAngles.fill(initAngle)
        # initial optimization attempt, in order to find minimal amount of plies
        solution = optimizeAngles(numLayers, initialSymAngles, 1000, knockDown, maxHalfwaves)
        print(numLayers, solution.x, solution.fun)
        if solution.fun < 1/minRF:
            optimalSymLayers = numLayers
            optimalSymAngles = solution.x
            print(np.concatenate((optimalSymAngles, optimalSymAngles[::-1])))
            print(optimalSymLayers * 2)
            print(1 / solution.fun)

            # second optimization step in order to maximize strength of found minimal layer count
            print("Second Optimization Step")
            solution = optimizeAngles(optimalSymLayers, optimalSymAngles, 10000, knockDown, maxHalfwaves)
            optimalSymLayers = numLayers
            optimalSymAngles = solution.x
            print("A:")
            print(mf.A_mat_sym(optimalSymAngles))
            print("B:")
            print(mf.B_mat_sym(optimalSymAngles))
            print("D:")
            print(mf.D_mat_sym(optimalSymAngles))
            print(np.concatenate((optimalSymAngles, optimalSymAngles[::-1])))
            print(optimalSymLayers * 2)
            print(1 / solution.fun)

            return solution
