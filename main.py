import GradientOptimize
import PermutationSearch

# Material
E11 = 132741.56  # 130_000 to 160_000 MPa
E22 = 10210.89  # 10_000 to 14_000 MPa
G12 = 5105.44  # 5_000 to 8_000 MPa
tLayer = 0.184  # 0.184mm to 0.25mm
a = 400  # 400mm-800mm
b = 250  # 150mm-250mm
alpha = a / b

knockDown = 0.9

# Loads
N_x = -1000  # -500N/mm to -2000N/mm
N_y = -500  # 300N/mm to -1000N/mm
Tau = 100  # 100 to 800N/mm
beta = N_y / N_x


def main():
    PermutationSearch.findOptSequence(24, 100, 20, 10000, 2, knockDown, 3)
    #GradientOptimize.optimalLayers(45, 10, 44, 2, knockDown, 3)


if __name__ == "__main__":
    main()
