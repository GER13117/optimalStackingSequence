import time
from math import factorial

import MaterialFunc as mf
import numpy as np
import random


# explained by https://www.sciencedirect.com/science/article/pii/S0263822314005492

def progress_bar(counter, max_value, bar_length=40, additional_value=None):
    progress = counter / max_value
    block = int(round(bar_length * progress))
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
        numbers.append(round(current, 2))
        current += interval
    return numbers


def checkPlyShare(stackingSeq: list):
    dedupSeq = list(dict.fromkeys(stackingSeq))
    shareArray = []
    for angle in dedupSeq:
        plyShare = stackingSeq.count(angle) / len(stackingSeq)
        shareArray.append(plyShare)
    return max(shareArray) <= 0.1


def generateRandomStackPairs(availableAngles, numSymAngles):
    stackingSeq = [random.choice(availableAngles) for _ in range(int(numSymAngles / 2))]
    negAngles = [-x for x in stackingSeq]
    stackingSeq.extend(negAngles)
    if numSymAngles % 2 != 0:
        stackingSeq.append(random.choice([0.0, 90.0]))
    stackingSeq = [90.0 if angle == -90.0 else angle for angle in stackingSeq]
    return stackingSeq


def generatStacksWithMidP(halfStack: list,
                          midPLaneAngle: float):  # Takes a half of a stacking sequence and returns two whole stacking sequences with 0° and 90° as its midplane
    fullStack = halfStack.copy()
    fullStack.append(midPLaneAngle)
    fullStack.extend(halfStack[::-1])

    return fullStack


def stackIsBalanced(stackingSeq):
    m_matA = mf.A_mat_sym(stackingSeq)
    return np.isclose(m_matA[0][2], 0, atol=1e-10) and np.isclose(m_matA[1][2], 0, atol=1e-10)

def stackIsBalancedMidP(stackingSeq):
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


def stackingSeqFromList(symSeq: list, even: bool):
    sequence_str = '/'.join(f'{angle}' for angle in symSeq)
    sequence_str = f'[{sequence_str}]'
    if even:
        sequence_str += 's'
    return sequence_str


def printResultMinimal(result: dict):
    print("Number auf Layers:", result["numLayers"],
          "\nRF:", result["RF"])


def printResultFull(result: dict):
    print("Number auf Layers:", result["numLayers"])
    if result["midPAngle"] is None:
        print("Best Stacking Sequence:", stackingSeqFromList(result["bestStackingSeq"], True))
    else:
        print("Best Stacking Sequence:",
              stackingSeqFromList(generatStacksWithMidP(result["bestStackingSeq"], result["midPAngle"]), False))
    print("R:", result["R"], "RF:", result["RF"],
          "\nis balanced:", result["isBalanced"],
          "- has max 10% plyshare:", result["hasMax10ppPlyShare"])


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
        bestCurrentR = mf.R_panelbuckling_comb_sym(stackingSeqRev[::-1], knockDown, maxHalfwaves)
        bestCurrentSeqRev = stackingSeqRev[:]
        progress_bar(idxSeq + 1, len(stackingSeqsRev), additional_value=f"Best RF: {1 / bestR:.3f}")
        for i in range(numSymLayers):
            for j in range(i + 1, numSymLayers):
                swappedStackingSeqRev = swapValues(stackingSeqRev, i, j)
                if swappedStackingSeqRev == stackingSeqRev:
                    continue
                stackingSeqRev = swappedStackingSeqRev
                R = mf.R_panelbuckling_comb_sym(stackingSeqRev[::-1], knockDown, maxHalfwaves)
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
        "midPAngle": None,
        "isBalanced": stackIsBalanced(bestStackingSeq),
        "hasMax10ppPlyShare": checkPlyShare(bestStackingSeq)
    }
    return result


def optimizeLayersMidP(numSymLayers: int, numStackingSeqs: int, interval: float, knockDown: float, maxHalfwaves: int):
    stackingSeqsRev = generateStackingSeqs(numSymLayers, numStackingSeqs, interval)
    bestR = 10000
    bestStackingSeqRev = stackingSeqsRev[0]
    bestMidPAngle = 0.0

    print("Optimizing Stacking Sequence:")
    for idxSeq, stackingSeqRev in enumerate(stackingSeqsRev):
        bestCurrentR90 = mf.R_panelbuckling_comb(generatStacksWithMidP(stackingSeqRev[::-1], 90.0), knockDown,
                                                 maxHalfwaves)
        bestCurrentR0 = mf.R_panelbuckling_comb(generatStacksWithMidP(stackingSeqRev[::-1], 0.0), knockDown,
                                                maxHalfwaves)
        if bestCurrentR90 > bestCurrentR0:
            bestCurrentR = bestCurrentR90
            bestMidPAngle = 90.0
        else:
            bestCurrentR = bestCurrentR0
            bestMidPAngle = 0.0

        bestCurrentSeqRev = stackingSeqRev[:]
        bestCurrentMidPAngle = 0.0
        progress_bar(idxSeq + 1, len(stackingSeqsRev), additional_value=f"Best RF: {1 / bestR:.3f}")
        for i in range(numSymLayers):
            for j in range(i + 1, numSymLayers):
                swappedStackingSeqRev = swapValues(stackingSeqRev, i, j)
                if swappedStackingSeqRev == stackingSeqRev:
                    continue
                stackingSeqRev = swappedStackingSeqRev

                R90 = mf.R_panelbuckling_comb(generatStacksWithMidP(stackingSeqRev[::-1], 90.0), knockDown,
                                              maxHalfwaves)
                R0 = mf.R_panelbuckling_comb(generatStacksWithMidP(stackingSeqRev[::-1], 0.0), knockDown, maxHalfwaves)
                if bestCurrentR90 > bestCurrentR0:
                    R = R90
                    midPAngle = 90.0
                else:
                    R = R0
                    midPAngle = 0.0

                if R < bestCurrentR:
                    bestCurrentR = R
                    bestCurrentMidPAngle = midPAngle
                    bestCurrentSeqRev = stackingSeqRev[:]
                    if bestCurrentR < bestR:
                        bestR = bestCurrentR
                        bestMidPAngle = bestCurrentMidPAngle
                        bestStackingSeqRev = bestCurrentSeqRev[:]
            stackingSeqRev = bestCurrentSeqRev[:]
    print(" ")
    bestStackingSeq = bestStackingSeqRev[::-1]
    result = {
        "numLayers": numSymLayers * 2 + 1,
        "bestStackingSeq": bestStackingSeq,
        "midPAngle": bestMidPAngle,
        "R": round(bestR, 3),
        "RF": round(1 / bestR, 3),
        "isBalanced": stackIsBalancedMidP(generatStacksWithMidP(bestStackingSeq, bestMidPAngle)),
        "hasMax10ppPlyShare": checkPlyShare(generatStacksWithMidP(bestStackingSeq, bestMidPAngle))
    }
    return result


def checkInputs(minLayers: int, maxLayers: int, interval: float):
    if minLayers < 20:
        print(
            "WARNING: For the given minimum layer count a maximum 10% plyshare is impossible. The minimal ply amount is 20. The minimum plycount will be set to 20")

    if maxLayers < 20:
        print(
            "WARNING: For the given maximum layer count a maximum 10% plyshare is impossible. The minimal ply amount is 20. The maximum plycount will be set to 20")

    minDifferentAngles = int(maxLayers / 2 / 2)  # minimum amount of different angles
    maxPossibleAngles = 180 / interval  # max amount of possible different angles
    if maxPossibleAngles < minDifferentAngles:
        print("WARNING: The given angle interval of", interval,
              "is too big for your maximum Layer amount. The program will get stuck at", 360 / interval, "Layers")

    time.sleep(1.0)


def findOptSequence(minLayers: int, maxLayers: int, popSizeCoarse: int, popSizeFine: int, interval: float,
                    knockDown: float, maxHalfwaves: int, checkWithMidP: bool, minRF=1.0):
    if maxLayers < minLayers:
        print("max. and min. layers are switched")
        minLayers, maxLayers = maxLayers, minLayers

    checkInputs(minLayers, maxLayers, interval)

    minLayers = max(minLayers, 20)
    maxLayers = max(maxLayers, 20)

    for i in range(minLayers, maxLayers + 1):
        if i % 2 != 0 and not checkWithMidP:
            continue
        print("<--------------------------------------------->")
        coarseResult = optimizeLayers(int(i / 2), popSizeCoarse, interval, knockDown, maxHalfwaves) if i % 2 == 0 \
            else optimizeLayersMidP(int(i / 2), popSizeCoarse, interval, knockDown, maxHalfwaves)
        printResultMinimal(coarseResult)
        if coarseResult["RF"] >= minRF:
            print("<=============================================>")
            print("Found first Solution - Starting second optimization Step")
            fineResult = optimizeLayers(int(i / 2), popSizeFine, interval, knockDown, maxHalfwaves) if i % 2 == 0 \
                else optimizeLayersMidP(int(i / 2), popSizeFine, interval, knockDown, maxHalfwaves)
            if fineResult["RF"] < coarseResult["RF"]:  # For the unlikely case that no better solution is found in the second optimization step
                printResultFull(coarseResult)
            else:
                printResultFull(fineResult)
            break
