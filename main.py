from scipy.optimize import minimize
import numpy as np
# Einschränkungen: nur symmetrische Laminate mit gerader Anzahl von schichten

def layerCount(layerCount):
    return layerCount


# constraint 1 - ineq
def RF_panelbuckling(x, numLayers, appliedLoad):
    layerAngles = x + x[::-1]

    #do shit


# constrain 2 - eq
def balanced(x):
    return sum(x*2) #nicht richtig


appliedLoad = (16, 15)


def optimizeAngles(numLayers):
    initialAngles = np.zeros(numLayers)

    # Define bounds for angles
    angleBounds = [(-90, 90) for _ in range(numLayers)]

    con1 = {'type': 'ineq', 'fun': lambda x: RF_panelbuckling(x, numLayers, appliedLoad)}
    con2 = {'type': 'eq', 'fun': balanced}
    cons = [con1, con2]

    solution = minimize(layerCount, initialAngles, method='SLSQP', bounds=angleBounds, constraints=cons)
    return solution


minLayers = 1
maxLayers = 50
optimalLayers = maxLayers
optimalAngles = []

for numLayers in range(minLayers, maxLayers + 1):
    solution = optimizeAngles(numLayers)

    if solution.success:
        optimalLayers = numLayers
        optimalAngles = solution.x

print("Done")