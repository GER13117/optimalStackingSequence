import MaterialFunc as mf
import numpy as np
from scipy.optimize import minimize
import random

possibleAngles = [90, 0, 45, -45]


def generateRandomStack(m_availableAngles, m_numSymAngles):
    stackingSeq = [random.choice(m_availableAngles) for _ in range(m_numSymAngles)]
    return stackingSeq


def generateStackingSeqs(numSymLayers, numStackingSeqs):
    m_initialSequences = []
    while len(m_initialSequences) < numStackingSeqs:
        m_possibleStackingSeq = generateRandomStack(possibleAngles, numSymLayers)
        m_matA = mf.A_mat(m_possibleStackingSeq)
        if np.isclose(m_matA[0][2], 0, atol=1e-10) and np.isclose(m_matA[1][2], 0, atol=1e-10):
            m_initialSequences.append(m_possibleStackingSeq)
    return m_initialSequences


def swapValues(sequence: list, idx1: int, idx2: int):
    sequence[idx1], sequence[idx2] = sequence[idx2], sequence[idx1]
    return sequence


def optimizeLayers(numSymLayers, numStackingSeqs):
    stackingSeqs = generateStackingSeqs(numSymLayers, numStackingSeqs)
    for stackingSeq in stackingSeqs:
        m_matD = mf.D_mat(stackingSeq)
        m_D11 = m_matD[0][0]
        m_D12 = m_matD[0][1]
        m_D22 = m_matD[1][1]
        m_D66 = m_matD[2][2]
        mf.sig_x_cr_biax()
