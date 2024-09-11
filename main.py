import GradientOptimize
import PermutationSearch
N_x = -1000  # -500N to -2000N
N_y = -500  # 300N to -1000N
Tau = 100  # 100 to 800N
beta = N_y / N_x

a = 400  # 400mm-800mm
b = 250  # 150mm-250mm
alpha = a / b
tLayer = 0.184  # 0.184mm to 0.25mm

def main():
    #GradientOptimize.optimalLayers(45, 44, 44, 2, N_x, Tau, beta, b, alpha)
    PermutationSearch.optimizeLayers(22, 1000, alpha, b, beta, tLayer, N_x, Tau, 5)


if __name__ == "__main__":
    main()