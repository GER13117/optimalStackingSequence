# Stacking Sequence Optimizer
This program optimizes stacking sequences for rectangular, simply supported, symmetrical and balanced composite layups with a maximum plyshare of 10%. 
It works by finding the minimal number of plies for which the RF exceeds a specified threshold (default: 1) and then optimizing the ply count for maximum strength.
## Basic usage
The user sets the material constants (E11, E12, G12, thickness, width and length) and the applied loads (N_x, N_y, Tau). \
Two optimization algorithms are available: Gradient Optimize and Permutation Search. All solvers and constants are set in the main.py file.
### Permutation Search
Permutation search works by generating an initial set (population) of pseudo random stacking sequences. 
By switching the layers within each sequence it optimizes each sequence for strength. In the end the sequence with the highest RF is returned. \
It operates in two steps: First, a coarse optimization with a small population which runs until it finds a minimum amount of layers for which a RF is reached. 
The second step uses a much larger population and runs for the minimum layer count determined in the previous step.

**The following parameters are available:**
* `minLayers` -  The minimum number of layers which will be checked by the solver
* `maxLayers` - The maximum number of layers which will be checked by the solver
* `popSizeCoarse` - The population size for the first optimization step. (For `interval` > 1° a value between 20 and 50 works fine, higher values slow down the first step)
* `popSizeFine` - The population size for the second optimization step. The default value is 5000. Higher values do not seem to improve the result
* `interval` - The interval for which possible angles of each layer are generated in degrees (i.e.: `interval=45` => possible angles `[-45°, 0°, 45°, 90°]`)
* `knockDown` - a knockdown value for D11, D12, D22, D66. Default is 0.9
* `maxHalfwaves` - the maximum amount of halfwaves the plane could have. the solver automatically chooses the most unstable one. Default is 3
* `checkWithMidP` - This parameter determines whether stacking sequences with an odd number of plies are considered.
  * False: Only stacking sequences with an even amount of plies are considered (i.e.: [90/45/-45]s)
  * True: stacking sequences with and odd amount of plies are also considered (i.e.: [90/45/-45/0/-45/45/90]). 
    As the stack has to be balanced, the innermost ply can only have the values 0° and 90°.
  * Enabling this function significantly slows down the solving process. It is recommended to only enable it after an optimal, even sequence is found and then to check its "neighbours"
* `minRF` - The minimal RF that causes to the optimizer to switch into "fine" mode. Default is 1.0

### Gradient Optimize
This algorithm leverages the scipy.minimize function to find the optimal stacking sequence.
It has a similar two-step approach as *Permutation Search*. 
However, this algorithm is less customizable and has less features than Permutation Search:
- it may produce angles with many decimal places, complicating manufacturing. 
- it is not ensured that the final result has a maximum ply-share of 10%. 
- only plies with an even layer count can be optimized

Also, the algorithm is a lot slower than Permutation search. \
In general you should **NOT** use this algorithm as permutation search is superior in most scenarios.

**The following parameters are available:**
* `initAngle` - used to generate an initial stacking sequence which will be optimized. Vary this variable to avoid local minima/maxima.
* `minLayers` -  The minimum number of layers which will be checked by the solver
* `maxLayers` - The maximum number of layers which will be checked by the solver
* `maxDecimals` - depreceated
* `knockDown` - a knockdown value for D11, D12, D22, D66. Default is 0.9
* `maxHalfwaves` - the maximum amount of halfwaves the plane could have. the solver automatically chooses the most unstable one. Default is 3
* `minRF` - The minimal RF that causes to the optimizer to switch into "fine" mode. Default is 1.0

## Dependencies
Both algorithms are dependent on `numpy`. *Gradient Optimize* additionally depends on `scipy`