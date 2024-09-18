import MaterialFunc as mf
import numpy as np
from scipy.optimize import minimize

def optimizeAngles(numLayers, initialSymAngles, iterCount, maxDecimals, knockDown, maxHalfwaves):
    # Define bounds for angles
    m_angleBounds = [(-90, 90) for _ in range(numLayers)]

    # BALANCED PLY
    m_con1 = {'type': 'eq', 'fun': lambda sym_angles: mf.A_mat_sym(sym_angles)[0][2]} # TODO: Fix Sym
    m_con2 = {'type': 'eq', 'fun': lambda sym_angles: mf.A_mat_sym(sym_angles)[1][2]} # TODO: Fix Sym

    # PLY DOES NOT BUCKLE
    m_con3 = {'type': 'ineq',
              'fun': lambda sym_angles: 1 - mf.R_panelbuckling_comb_sym(sym_angles, knockDown, maxHalfwaves)} # TODO: Fix Sym

    # ANGLES ONLY HAVE SOME AMOUNT OF DECIMALS TODO: FIX NEEDED
    m_con4 = {'type': 'eq',
              'fun': lambda sym_angles: np.amax(abs(np.around(sym_angles, decimals=maxDecimals) - sym_angles))}

    # TODO: Add 10% ply share constraint
    # m_con6 = "10% of each ply share"
    m_cons = [m_con1, m_con2, m_con3]

    m_options = {'maxiter': iterCount}
    m_optimal_R = minimize(mf.R_panelbuckling_comb_sym, initialSymAngles, method='trust-constr', bounds=m_angleBounds,  # TODO: Fix Sym
                           constraints=m_cons,
                           options=m_options,
                           args=(knockDown, maxHalfwaves))
    return m_optimal_R

def optimalLayers(initAngle: float, minLayers: int, maxLayers: int, maxDecimals: int, knockDown: float, maxHalfwaves: int):
    for numLayers in range(int(minLayers / 2), int(maxLayers / 2) + 1):
        initialSymAngles = np.empty(numLayers)
        initialSymAngles.fill(initAngle)
        # initial optimization attempt, in order to find minimal amount of plies
        solution = optimizeAngles(numLayers, initialSymAngles, 1000, maxDecimals, knockDown, maxHalfwaves)
        print(numLayers, solution.x, solution.fun)
        if solution.fun < 1:  # TODO: Experiment with different values close to 1
            optimalSymLayers = numLayers
            optimalSymAngles = solution.x
            print(np.concatenate((optimalSymAngles, optimalSymAngles[::-1])))
            print(optimalSymLayers * 2)
            print(1 / solution.fun)

            # second optimization step in order to maximize strength of found minimal layer count
            print("Second Optimization Step")
            solution = optimizeAngles(optimalSymLayers, optimalSymAngles, 10000, maxDecimals, knockDown, maxHalfwaves)
            optimalSymLayers = numLayers
            optimalSymAngles = solution.x
            print("A:")
            print(mf.A_mat_sym(optimalSymAngles)) # TODO: Fix Sym
            print("B:")
            print(mf.B_mat_sym(optimalSymAngles)) # TODO: Fix Sym
            print("D:")
            print(mf.D_mat_sym(optimalSymAngles)) # TODO: Fix Sym
            print(np.concatenate((optimalSymAngles, optimalSymAngles[::-1])))
            print(optimalSymLayers * 2)
            print(1 / solution.fun)

            return solution
