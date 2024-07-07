import numpy as np

from scipy import optimize
from scipy.optimize import leastsq
from scipy import interpolate


def lorentz(p, x):
    return p[0] / ((x - p[1]) ** 2 + p[2])


def errorfunc(p, x, z):
    return z - lorentz(p, x)


# 拟合函数
def lorentz_fit(x, y):
    # x表示频率，y表示对应能量
    p3 = ((max(x) - min(x)) / 10) ** 2
    p2 = (max(x) + min(x)) / 2
    p1 = max(y) * p3
    c = np.min(y)
    p0 = np.array([p1, p2, p3, c], dtype=np.float)  # Initial guess
    solp, ier = leastsq(errorfunc, p0, args=(x, y), maxfev=200000)
    return lorentz(solp, x), solp


# 中心频率-温度转换关系
def get_temp_BFS(BFS):
    # BFS->Temperature,对于不同光纤参数不同
    T_RawData = (BFS - 10.802052311464523) / 0.001034287825626
    return T_RawData


# RMSE
def rmse(y, y_):
    # 一个是真实值，一个是测量值，计算结果是测量值与真实值的误差
    RMSE = np.sqrt(np.mean((y - y_) ** 2))
    return RMSE


# SD
def sd(y):
    # 描述数据的波动程度
    SD = np.sqrt(np.var(y))
    return SD


# SNR
def snr(y, y_lortz, solp):
    # y是被测数据，y_lortz是y经过Lorentz拟合的结果，solp是拟合后的参数
    y_lortz_max = lorentz(solp, solp[1])
    var = np.var(y - y_lortz)
    SNR = y_lortz_max ** 2 / var
    SNR_dB = 10 * np.log10(SNR)
    return SNR_dB


# 空间分辨率
def get_SR(data, SPN=10):
    def f(x, A, B):  # this is your 'straight line':y=f(x)
        return A * x + B

    startpos1 = np.int(data.shape[0] * 840 * (SPN / 10))
    endpos1 = np.int(data.shape[0] * 1340 * (SPN / 10))

    startpos2 = np.int(data.shape[0] * 1650 * (SPN / 10))
    endpos2 = np.int(data.shape[0] * 2150 * (SPN / 10))

    shiftBefore = np.average(data[0][startpos1:endpos1])
    shiftAfter = np.average(data[0][startpos2:endpos2])

    # 拟合上升沿直线 y=kx+b
    shiftpos = np.array([int(1588 * (SPN / 10)), int(1590 * (SPN / 10)), int(1592 * (SPN / 10)), int(1594 * (SPN / 10)),
                         int(1596 * (SPN / 10))])

    y = np.array(data[0][shiftpos])
    A, B = optimize.curve_fit(f, shiftpos, y)[0]

    start_loc = (shiftBefore - B) * .2 / A
    end_loc = (shiftAfter - B) * .2 / A

    # 空间分辨率
    SR = ((end_loc - start_loc) * 0.8) / SPN
    return SR


# 重采样
def ResamplingBGS(Data, SPNOriginal, SPNResample, Method='linear'):
    Data = np.array(Data)
    Height, Width = Data.shape

    x = np.linspace(0, Width - 1, Width)
    NewWidth = int(Width * (SPNResample / SPNOriginal))
    xNew = np.linspace(0, Width - 1, NewWidth)
    DataNew = np.zeros((Height, NewWidth))
    for i in range(0, Height - 1):
        ResamplingFunction = interpolate.interp1d(x, Data[i, :], kind=Method)
        yNew = np.array(ResamplingFunction(xNew))
        DataNew[i, :] = yNew
    return DataNew


# 半高全宽
def FullWidthHalfHeight(Data, Standard):
    PeakData = Data[1510:1540]
    PeakLeft = Data[1510:1525]
    PeakRight = Data[1525:1540]

    Max = np.max(PeakData)
    Mid = ((Max - Standard) / 2) + Standard

    PeakLeftUpper = PeakLeft[PeakLeft >= Mid]
    PeakLeftLower = PeakLeft[PeakLeft < Mid]
    PeakLeftUpperLabel = np.where(Data == np.min(PeakLeftUpper))[0]
    PeakLeftLowerLabel = np.where(Data == np.max(PeakLeftLower))[0]

    PeakRightUpper = PeakRight[PeakRight >= Mid]
    PeakRightLower = PeakRight[PeakRight < Mid]
    PeakRightUpperLabel = np.where(Data == np.min(PeakRightUpper))[0]
    PeakRightLowerLabel = np.where(Data == np.max(PeakRightLower))[0]

    Width = (abs(PeakLeftUpperLabel - PeakRightUpperLabel)[0] + abs(PeakLeftLowerLabel - PeakRightLowerLabel)[0]) / 2
    return Width


# 得出BFS和SNR
def BGS2BFS(BGS, FrequencyMax=10.950, FrequencyMin=10.750, FreqPointNumber=201):
    FrequencyList = np.linspace(FrequencyMin, FrequencyMax, FreqPointNumber)
    BGS = np.array(BGS).astype(np.float32)

    SNRBGS = []
    LortzBGS = []
    for each in BGS:
        lortz, solp = lorentz_fit(FrequencyList, each)
        SNRBGS.append(snr(each, lortz, solp))
        LortzBGS.append(solp[1])
    BFS = np.array(LortzBGS)
    Temperature = get_temp_BFS(BFS)
    SNRBGS_filter1 = np.array(SNRBGS)
    SNR = np.average(SNRBGS_filter1)
    return BFS, Temperature, SNR
