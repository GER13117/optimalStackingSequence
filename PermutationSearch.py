import MaterialFunc as mf
import numpy as np
from scipy.optimize import minimize
import random

possibleAngles = [90, 0, 45, -45]


def generateRandomStack(m_availableAngles, m_numSymAngles):
    stackingSeq = [random.choice(m_availableAngles) for _ in range(m_numSymAngles)]
    return stackingSeq


def stackIsBalanced(stackingSeq):
    m_matA = mf.A_mat(stackingSeq)
    return np.isclose(m_matA[0][2], 0, atol=1e-10) and np.isclose(m_matA[1][2], 0, atol=1e-10)


def generateStackingSeqs(numSymLayers, numStackingSeqs):
    m_initialSequences = []
    while len(m_initialSequences) < numStackingSeqs:
        m_possibleStackingSeq = generateRandomStack(possibleAngles, numSymLayers)
        if stackIsBalanced(m_possibleStackingSeq):
            m_initialSequences.append(m_possibleStackingSeq)
    return m_initialSequences


def swapValues(sequence: list, idx1: int, idx2: int):
    newSeq = sequence[:]
    newSeq[idx1], newSeq[idx2] = newSeq[idx2], newSeq[idx1]
    return newSeq


def R_panelbuckling_comb(stackingSeq: list, alpha, b, beta, tLayer, N_x, Tau):
    m_matD = mf.D_mat(stackingSeq)
    m_D11 = m_matD[0][0]
    m_D12 = m_matD[0][1]
    m_D22 = m_matD[1][1]
    m_D66 = m_matD[2][2]
    m, n = 1, 1  # TODO: minimizer
    m_hPanel = tLayer * len(stackingSeq)
    m_sigma_x_cr = mf.sig_x_cr_biax(b, m_hPanel, beta, alpha, m, n, m_D11, m_D12, m_D22, m_D66)
    m_tau_cr = mf.tau_cr(b, m_hPanel, m_D11, m_D12, m_D22, m_D66)

    # applied loads
    m_sigma_x = N_x / m_hPanel
    m_tau = Tau / m_hPanel

    # R values
    m_R_biax = abs(m_sigma_x / m_sigma_x_cr)
    m_R_shear = abs(m_tau / m_tau_cr)

    m_R = m_R_biax + m_R_shear ** 2

    return m_R


def optimizeLayers(numSymLayers: int, numStackingSeqs: int, alpha, b, beta, tLayer, N_x, Tau):
    stackingSeqsRev = generateStackingSeqs(numSymLayers, numStackingSeqs)
    bestR = 10000
    bestStackingSeqRev = stackingSeqsRev[0]
    for stackingSeqRev in stackingSeqsRev:
        bestCurrentR = R_panelbuckling_comb(stackingSeqRev, alpha, b, beta, tLayer, N_x, Tau)
        bestCurrentSeqRev = stackingSeqRev[:]
        for i in range(numSymLayers):
            for j in range(i + 1, numSymLayers):
                swappedStackingSeqRev = swapValues(stackingSeqRev, i, j)
                if swappedStackingSeqRev == stackingSeqRev:
                    continue
                stackingSeqRev = swappedStackingSeqRev
                R = R_panelbuckling_comb(stackingSeqRev[::-1], alpha, b, beta, tLayer, N_x, Tau)
                if R < bestCurrentR:
                    bestCurrentR = R
                    bestCurrentSeqRev = stackingSeqRev[:]
                    if bestCurrentR < bestR:
                        bestR = bestCurrentR
                        bestStackingSeqRev = bestCurrentSeqRev[:]
                #print("i:", i, "stacking Sequence:", stackingSeqRev[::-1], "R:", R)
            stackingSeqRev = bestCurrentSeqRev[:]
            #print("i:", i, "best current Sequence:", stackingSeqRev[::-1], "R:", bestCurrentR)
        print("best final Sequence:", stackingSeqRev[::-1], "R:", bestCurrentR, "is balanced:", stackIsBalanced(stackingSeqRev[::-1]))

    print("Best Stacking Sequence:", bestStackingSeqRev[::-1], "R:", bestR)
