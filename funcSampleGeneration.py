import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi
from scipy.signal import convolve
from numpy.random import randn
import random

def awgn(signal, snr_dB):
    snr = 10**(snr_dB / 10.0)
    power_signal = np.sum(np.abs(signal)**2) / signal.size
    power_noise = power_signal / snr
    noise = np.sqrt(power_noise) * (randn(signal.size) + 1j * randn(signal.size)) / np.sqrt(2)
    return signal + noise, noise

def gmsk_modulate(bits, bt_product, samples_per_symbol):
    h = 0.5
    g = np.sinc(np.arange(-4, 4 + 1 / samples_per_symbol, 1 / samples_per_symbol)) * np.hamming(8 * samples_per_symbol + 1)
    g = g / np.sum(g)
    freq = convolve(np.repeat(bits, samples_per_symbol), g, mode='same')
    phase = np.cumsum(freq * pi * h)
    return np.exp(1j * phase)

def generate_image(signal, consScaleQ,consScaleI,dQX,dIY,dXY,imageSizeX,maxBlkSize,imageSizeY,blkSize,pixelCentroid,cFactor,imageDir,imageName,txtPath,setPath,genType): 

    sampleX = np.fix((consScaleQ - np.imag(signal)) / dQX).astype(int)
    sampleY = np.fix((consScaleI + np.real(signal)) / dIY).astype(int)

    imageArray = np.zeros((imageSizeX + 2 * maxBlkSize, imageSizeY + 2 * maxBlkSize, 3))

    for kk in range(3):
        blkXmin = sampleX - blkSize[kk]
        blkXmax = sampleX + blkSize[kk]
        blkYmin = sampleY - blkSize[kk]
        blkYmax = sampleY + blkSize[kk]

        for ii in np.where((blkXmin > 0) & (blkYmin > 0) & (blkXmax < imageSizeX) & (blkYmax < imageSizeY))[0]:
            sampleDistance = np.abs(signal[ii] - pixelCentroid[blkXmin[ii]:blkXmax[ii], blkYmin[ii]:blkYmax[ii]])
            imageArray[blkXmin[ii]:blkXmax[ii], blkYmin[ii]:blkYmax[ii], kk] += np.exp(-cFactor[kk] * sampleDistance / dXY)

        imageArray[:, :, kk] /= np.max(imageArray[:, :, kk])

    plt.imsave(os.path.join(imageDir, f"{imageName}.png"), imageArray[(2 * maxBlkSize):(imageSizeX - 2 * maxBlkSize), (2 * maxBlkSize):(imageSizeY - 2 * maxBlkSize), :])

    print(f"{txtPath}{imageName}.png \n")
    with open(f"./{setPath}/{genType}.txt", 'a') as file:
        file.write(f"{txtPath}{imageName}.png \n")

    return

def generate_constellation_images(modType, samplesPerImage, imageNum, imageSize, consScale, set_type, setPath):

    blkSize = [5, 25, 50]
    cFactor = 5.0 / np.array(blkSize)

    snr_ul = 10
    snr_ll = 10
    SNR_dB = round(random.uniform(-snr_ll, snr_ul), 2) # TODO: needs to fix. Has to be the same for all the modality

    dIY = 2 * consScale[0] / imageSize[0]
    dQX = 2 * consScale[1] / imageSize[1]
    dXY = np.sqrt(dIY**2 + dQX**2)

    maxBlkSize = max(blkSize)
    imageSizeX = imageSize[0] + 4 * maxBlkSize
    imageSizeY = imageSize[1] + 4 * maxBlkSize

    consScaleI = consScale[0] + 2 * maxBlkSize * dIY
    consScaleQ = consScale[1] + 2 * maxBlkSize * dQX

    if modType == 'OOK':
        consDiag = [0, 1]
        modOrder = 1
    elif modType == '4ASK':
        consDiag = [-3, -1, 1, 3]
        modOrder = 2
    elif modType == '8ASK':
        consDiag = [-7, -5, -3, -1, 1, 3, 5, 7]
        modOrder = 3
    elif modType == 'OQPSK':
        consDiag = np.exp((np.arange(4) / 4) * 2 * pi * 1j + pi / 4)
        modOrder = 2
    elif modType == 'CPFSK':
        consDiag = [np.exp(1j * 2 * pi * 0.25), np.exp(1j * 2 * pi * 0.75)]
        modOrder = 1
    elif modType == 'GFSK':
        consDiag = [np.exp(1j * 2 * pi * 0.25), np.exp(1j * 2 * pi * 0.75)]
        modOrder = 1
    elif modType == '4PAM':
        consDiag = [-3, -1, 1, 3]
        modOrder = 2
    elif modType == 'DQPSK':
        consDiag = np.exp(1j * np.array([0, pi/2, pi, -pi/2]))
        modOrder = 2
    elif modType == '16PAM':
        consDiag = [-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15]
        modOrder = 4
    elif modType == 'GMSK':
        btProduct = 0.3
        samplesPerSymbol = 8
        consDiag = None
        modOrder = 1
    else:
        raise ValueError('Unrecognized Modulation Type!')

    imageIDWidth = len(str(imageNum))
    imageIDPrefix = f"{modType}_{SNR_dB}dB__"
    txtPath = ''

    # if genType in ['val', 'train', 'test']:
    #     imageDir = os.path.join(setPath, genType, modType)
    #     imageDir = os.path.join(setPath, genType)
    #     txtPath = ''

    for genType in set_type:
        imageDir = os.path.join(setPath, genType)
        if not os.path.exists(imageDir):
            os.makedirs(imageDir)


    # elif genType in ['noiseLessImg', 'noisyImg', 'signal', 'noise']:
    #     imageDir = os.path.join(setPath, genType)
    #     txtPath = ''
    # else:
    #     raise ValueError('Wrong Generation Type! Must be train/val/test/unlabeled')


    # dict
    # model_type = {
    # "OOK":{
    # "consDiag": [0, 1],
    # "modOrder": 1
    # },
    # 
    # }
    # if modType:
        # consDiag, modOrder, modLabel = model_type[modType].

    
        pixelCentroid = np.zeros((imageSizeX, imageSizeY), dtype=complex)
        for ii in range(imageSizeX):
            for jj in range(imageSizeY):
                pixelCentroid[ii, jj] = (-consScaleI + dIY / 2 + (jj - 1) * dIY) + \
                                        1j * (consScaleQ - dQX / 2 - (ii - 1) * dQX)

        for jj in range(imageNum):
            if modType == 'GMSK':
                msgBits = np.random.randint(0, 2, samplesPerImage)
                modSignal = gmsk_modulate(msgBits, btProduct, samplesPerSymbol)
                signalTx = modSignal
            else:
                msg = np.random.randint(1, 2**modOrder + 1, samplesPerImage)
                signalTx = np.zeros(samplesPerImage, dtype=complex)
                for ii in range(1, 2**modOrder + 1):
                    signalTx += consDiag[ii - 1] * (msg == ii)

                if modType in ['BPSK', '4ASK']:
                    signalTx[0] += 1j * 1E-4
            
            imageID = str(jj).zfill(imageIDWidth)
            imageName = f"{imageIDPrefix}{imageID}"

            timingErrorStd = 0.0001
            frequencyErrorStd = 0.0001

            timingError = randn() * timingErrorStd
            frequencyError = randn() * frequencyErrorStd

            phaseOffset = np.arange(len(signalTx)) * frequencyError + timingError
            signalTx *= np.exp(1j * phaseOffset) # Noise less image

            # print(signalTx.shape)

            signalRx, noise = awgn(signalTx, SNR_dB) # Noisy image after adding noise

            # print("noise ", noise.shape)

           
            # if not print_once:
            if genType == set_type[0]:
                with open(f"./{setPath}/files.txt", 'a') as file:
                    file.write(f"{txtPath}{imageName}\n")
                    # print_once = True

            if genType == 'noiseLessImg':
                generate_image(signalTx, consScaleQ,consScaleI,dQX,dIY,dXY,imageSizeX,maxBlkSize,imageSizeY,blkSize,pixelCentroid,cFactor,imageDir,imageName,txtPath,setPath,genType)
            elif genType == 'noisyImg':
                generate_image(signalRx, consScaleQ,consScaleI,dQX,dIY,dXY,imageSizeX,maxBlkSize,imageSizeY,blkSize,pixelCentroid,cFactor,imageDir,imageName,txtPath,setPath,genType)
            elif genType == 'signal':
                with open(os.path.join(imageDir, f"{imageName}.npy"), 'wb') as f:
                    np.save(f, signalTx)
                with open(f"./{setPath}/{genType}.txt", 'a') as file:
                    file.write(f"{txtPath}{imageName}.png \n")
            elif genType == 'noise':
                with open(os.path.join(imageDir, f"{imageName}.npy"), 'wb') as f:
                    np.save(f, noise)
                with open(f"./{setPath}/{genType}.txt", 'a') as file:
                    file.write(f"{txtPath}{imageName}.png \n")


