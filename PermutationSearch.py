import MaterialFunc as mf
import numpy as np
import random

# explained by https://www.sciencedirect.com/science/article/pii/S0263822314005492

def progress_bar(counter, max_value, bar_length=40, additional_value=None):
    # Calculate progress as a percentage
    progress = counter / max_value
    # Calculate the number of hashes (#) in the bar
    block = int(round(bar_length * progress))
    # Create the progress bar string
    progress_bar = "#" * block + "-" * (bar_length - block)

    # Display the progress bar with the percentage
    if additional_value is not None:
        print(f"\r[{progress_bar}] {progress * 100:.2f}% - {additional_value}", end="")
    else:
        print(f"\r[{progress_bar}] {progress * 100:.2f}%", end="")


def geneatePossibleAngles(start: float, end: float, interval: float):
    numbers = []
    current = start
    while current <= end:
        numbers.append(round(current, 6))  # Adjust precision as needed
        current += interval
    return numbers


def checkPlyShare(stackingSeq: list):
    dedupSeq = list(dict.fromkeys(stackingSeq))
    shareArray = []
    for angle in dedupSeq:
        plyShare = stackingSeq.count(angle) / len(stackingSeq)
        shareArray.append(plyShare)
    return max(shareArray) <= 0.1


def generateRandomStack(m_availableAngles, m_numSymAngles):
    stackingSeq = [random.choice(m_availableAngles) for _ in range(m_numSymAngles)]
    stackingSeq = [90.0 if angle == -90.0 else angle for angle in stackingSeq]
    return stackingSeq


def generateRandomStackPairs(m_availableAngles, m_numSymAngles):
    stackingSeq = [random.choice(m_availableAngles) for _ in range(int(m_numSymAngles / 2))]
    negAngles = [-x for x in stackingSeq]
    stackingSeq.extend(negAngles)
    if m_numSymAngles % 2 != 0:
        stackingSeq.append(random.choice([0.0, 90.0]))
    stackingSeq = [90.0 if angle == -90.0 else angle for angle in stackingSeq]
    return stackingSeq


def stackIsBalanced(stackingSeq):
    m_matA = mf.A_mat(stackingSeq)
    return np.isclose(m_matA[0][2], 0, atol=1e-10) and np.isclose(m_matA[1][2], 0, atol=1e-10)


def generateStackingSeqs(numSymLayers: int, numStackingSeqs: int, interval: float):
    m_initialSequences = []
    print("Generating stacking sequences")
    m_possibleAngles = geneatePossibleAngles(-90.0, 90.0, interval)
    while len(m_initialSequences) < numStackingSeqs:
        m_possibleStackingSeq = generateRandomStackPairs(m_possibleAngles, numSymLayers)
        if stackIsBalanced(m_possibleStackingSeq) and not (
                m_possibleStackingSeq in m_initialSequences) and checkPlyShare(m_possibleStackingSeq):
            m_initialSequences.append(m_possibleStackingSeq)
        progress_bar(len(m_initialSequences), numStackingSeqs,
                     additional_value=f"{len(m_initialSequences)}/{numStackingSeqs}")
    print(" ")
    return m_initialSequences


def swapValues(sequence: list, idx1: int, idx2: int):
    newSeq = sequence[:]
    newSeq[idx1], newSeq[idx2] = newSeq[idx2], newSeq[idx1]
    return newSeq

def optimizeLayers(numSymLayers: int, numStackingSeqs: int, interval: float, knockDown: float, maxHalfwaves: int):
    stackingSeqsRev = generateStackingSeqs(numSymLayers, numStackingSeqs, interval)
    bestR = 10000
    bestStackingSeqRev = stackingSeqsRev[0]
    print("Optimizing Stacking Sequence:")
    for idxSeq, stackingSeqRev in enumerate(stackingSeqsRev):
        bestCurrentR = mf.R_panelbuckling_comb(stackingSeqRev[::-1], knockDown, maxHalfwaves)
        bestCurrentSeqRev = stackingSeqRev[:]
        progress_bar(idxSeq + 1, len(stackingSeqsRev), additional_value=f"Best RF: {1 / bestR:.3f}")
        for i in range(numSymLayers):
            for j in range(i + 1, numSymLayers):
                swappedStackingSeqRev = swapValues(stackingSeqRev, i, j)
                if swappedStackingSeqRev == stackingSeqRev:
                    continue
                stackingSeqRev = swappedStackingSeqRev
                R = mf.R_panelbuckling_comb(stackingSeqRev[::-1], knockDown, maxHalfwaves)
                if R < bestCurrentR:
                    bestCurrentR = R
                    bestCurrentSeqRev = stackingSeqRev[:]
                    if bestCurrentR < bestR:
                        bestR = bestCurrentR
                        bestStackingSeqRev = bestCurrentSeqRev[:]
            stackingSeqRev = bestCurrentSeqRev[:]

    print(" ")
    bestStackingSeq = bestStackingSeqRev[::-1]
    result = {
        "numLayers": numSymLayers * 2,
        "bestStackingSeq": bestStackingSeq,
        "R": round(bestR, 3),
        "RF": round(1 / bestR, 3),
        "isBalanced": stackIsBalanced(bestStackingSeq),
        "hasMax10ppPlyShare": checkPlyShare(bestStackingSeq)
    }
    return result


def findOptSequence(minLayers: int, maxLayers: int, popSizeCoarse: int, popSizeFine: int, interval: float, knockDown: float, maxHalfwaves: int):
    minSymLayers = int(minLayers / 2)
    maxSymLayers = int(maxLayers / 2)
    for i in range(minSymLayers, maxSymLayers + 1):
        coarseResult = optimizeLayers(i, popSizeCoarse, interval, knockDown, maxHalfwaves)
        print("Number auf Layers:", coarseResult["numLayers"],
              "\nBest Stacking Sequence:", coarseResult["bestStackingSeq"],
              "\nR:", coarseResult["R"], "RF:", coarseResult["RF"],
              "\nis balanced:", coarseResult["isBalanced"],
              "- has max 10% plyshare:", coarseResult["hasMax10ppPlyShare"])
        if coarseResult["RF"] > 1.0:
            print("Found first Solution - Starting second optimization Step")
            fineResult = optimizeLayers(i, popSizeFine, interval, knockDown, maxHalfwaves)
            print("Number auf Layers:", fineResult["numLayers"],
                  "\nBest Stacking Sequence:", fineResult["bestStackingSeq"],
                  "\nR:", fineResult["R"], "RF:", fineResult["RF"],
                  "\nis balanced:", fineResult["isBalanced"],
                  "- has max 10% plyshare:", fineResult["hasMax10ppPlyShare"])
            break;