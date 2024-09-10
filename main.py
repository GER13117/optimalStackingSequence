import GradientOptimize

N_x = -1000  # -500N to -2000N
N_y = -500  # 300N to -1000N
Tau = 100  # 100 to 800N
beta = N_y / N_x

GradientOptimize.optimalLayers(45, 44, 44, 2, N_x, Tau, beta)
