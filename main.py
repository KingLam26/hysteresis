"""
This python file contains the HystersisAnalysis class, which contains all the functions needed to 
import, process and analysis the CSV data file exported from the PicoScope software.
"""

# ===== import business ===== #
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Polygon
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.optimize import minimize
import pickle
import random


# ===== class ===== #
class HystersisAnalysis:
    def __init__(self, filePath, materialName, areaAnalysisMode):
        """ initialization function, automatically performs:
                - key parameter generation
                - CSV file data import
                - splicing of the data points into multiple hysteresis loops
                - calculates B-H area for each hysteresis loop and prints results for that

        Args:
            filePath (string): CSV data file
            materialName (string): "mild steel" or "Cu-Ni alloy" or "transformer iron"
            areaAnalysisMode (string): "shoelace" or "shapely", for calculating B-H loop area treated a a polygon made of (H,B) data points
        """

        # Invariant parameters
        self.Np = 400                       # primary coil loop count
        self.Ns = 500                       # secondary coil loop count

        self.Lp = 4.2e-2                    # primary coil length, meters
        self.LpError = 0.1e-2
        
        self.Rp = 2.1                        # input resistor, ohms
        self.RpError = 0.1

        self.Rs = 9949                       # shunt resistor, ohms
        self.RsError = 6
        self.C = 473.6e-9                    # capacitor, Farads
        self.CError = 0.2e-9

        # Material properties
        self.filePath = filePath
        self.materialName = materialName
        self.materialProperties()

        # Model parameters
        self.buffer = 0     # decides how much of the next hysteresis loop data point to include in the previous loop
        self.freq = 50      # integrator circuit frequency
        self.areaAnalysisMode = areaAnalysisMode

        # imports data
        self.df = self.importCSV()

        # splits into separate loops
        self.dataDict = self.splitRawData()

        # basic loop area analysis
        self.loopCount, self.BHAreas, self.avgBHAreas, self.stdBHAreas = self.basicLoopAreaAnalysis()
        print(f'loop count: {self.loopCount}')
        print(f'BHAreas: {self.BHAreas}')
        print(f'avgBHAreas: {self.avgBHAreas}')
        print(f'stdBHAreas: {self.stdBHAreas}')


    def materialProperties(self):
        """ based on the materialName input parameter in the def __init__, 
            sets corresponding material parameters
        """
        
        if self.materialName == 'mild steel':
            self.Ds = 3.15e-3                               # diameter of sample, meters
            self.DsError = 0.01e-3

            self.As = np.pi * (self.Ds / 2) ** 2            # meters squared, circular cross section
            self.AsError = 2 * (self.DsError / self.Ds) * self.As

        elif self.materialName == 'Cu-Ni alloy':
            self.Ds = 4.99e-3                               # diameter of sample, meters
            self.DsError = 0.01e-3
            self.As = np.pi * (self.Ds / 2) ** 2            # meters squared, circular cross section
            self.AsError = 2 * (self.DsError / self.Ds) * self.As

        elif self.materialName == 'transformer iron':
            self.Ls = 4.22e-3                               # meters, length (acutally width)
            self.LsError = 0.01e-3
            
            self.Ws = 0.61e-3                               # meters, width (actually thickness)
            self.WsError = 0.01e-4
            
            self.As = self.Ls * self.Ws                     # meters squared, rect. cross section
            self.AsError = self.As * ( ( ( self.LsError / self.Ls ) ** 2 + ( self.WsError / self.Ws ) ** 2 ) ** 0.5 )

            # Assumes 0.1% error in voltages
            self.HFracError = ( 0.001**2 + (self.LpError / self.Lp)**2 + (self.RpError / self.Rp)**2 ) ** 0.5
            self.BFracError = ( 0.001**2 + (self.RsError / self.Rs)**2 + (self.CError / self.C)**2 + (self.AsError / self.As)**2) ** 0.5


    def pickleDump(self, target, name):
        """ ouputs data (for subsequent functions) and 
            saves it as a pickle data file on the hard drive

        Args:
            target (dict): data (dictionary) to be pickle dumped
            name (string): name of the file to be saved
        """

        with open(name, 'wb') as f:
            pickle.dump(target, f)
        return
    

    def pickleOpen(self, name):
        """ opens and read pickle data file

        Args:
            name (string): name of the file to be saved

        Returns:
            data dict: data / contents of the file
        """

        with open(name, 'rb') as f:
            target = pickle.load(f)
        return target


    def importCSV(self):
        """ reads CSV data file, output from PicoScope - voltages and time

        Returns:
            dataframe: raw data
        """

        df = pd.read_csv(self.filePath)
        return df


    def splitRawData(self):
        """ splits raw data from PicoScope CSV data file into individual hysteresis loops

        Returns:
            dict of dict: raw voltage data, calculated B and H values for each hysteresis loop
        """

        timeList = np.array(self.df.iloc[2:,0], dtype = float) * 10**(-3)       # raw CSV data

        period = 1 / self.freq
        deltaT = ( timeList[-1] - timeList[0] ) / ( len(timeList) - 1 )

        bufferCount = round( period * self.buffer / deltaT / 100 )              # how much of next loop to include in the previous loop to avoid incomplete data loops, buffer = 0 is fine here

        t0 = timeList[0]
        splitPos = []
        splitPosBuffer = []
        
        t1 = t0 + period
        for i in range(len(timeList)):
            if timeList[i] > t1:
                splitPos.append(i)                 # the last element inside the cycle is i-1, but as V[a:b] exclude b we record i instead of i-1
                if i + bufferCount <= len(timeList):
                    splitPosBuffer.append(i+bufferCount)
                else:
                    splitPosBuffer.append(i)
                t0 = timeList[i]
                t1 = t0 + period

        splitPos.insert(0,0)
        Vx = np.array(self.df.iloc[2:,1], dtype = float)
        Vy = np.array(self.df.iloc[2:,2], dtype = float)
        
        dataDict = {}
        index = 0
        for i in range(len(splitPosBuffer)):
            a = splitPos[i]
            b = splitPosBuffer[i]
            tSplit = timeList[a:b]
            VxSplit = Vx[a:b]
            VySplit = Vy[a:b]
            HSplit = ( self.Np * VxSplit ) / ( self.Lp * self.Rp )
            BSplit = ( self.Rs * self.C * VySplit ) / ( self.Ns * self.As)
            dataDict[index] = {'t': tSplit, 'Vx': VxSplit, 'Vy': VySplit, 'H': HSplit, 'B': BSplit}
            index += 1
        
        # original combined, unsplit
        H = ( self.Np * Vx ) / ( self.Lp * self.Rp )
        B = ( self.Rs * self.C * Vy ) / ( self.Ns * self.As)
        dataDict['combined'] = {'t': timeList, 'Vx': Vx, 'Vy': Vy, 'H': H, 'B': B}

        return dataDict


    def plotHysteresis(self, outputMode, combined):
        """ plots the hysteresis loops using Matplotlib

        Args:
            outputMode (string): "save" the output figures to hard drive or "show" but not save
            combined (boolean): if true, all the data points in the raw data file are plotted, i.e., 
                                rather than plotting only a single hysteresis loop that is the result of splitting the full data set
        """

        indexes = np.arange(len(self.dataDict.keys()))
        count = len(indexes) - 1        # last one is average, so minus 1
        if combined == False:
            for index in range(self.loopCount - 1):
                keys = self.dataDict[index].keys()
                if 'Vx' in keys:
                    fig = plt.figure(figsize = (10, 5))
                    fig.suptitle(f'Single hysteresis loop for {self.materialName}, index: {index}')
                    Vx = self.dataDict[index]['Vx']
                    Vy = self.dataDict[index]['Vy']
                    ax1 = fig.add_subplot(121)
                    ax1.scatter(Vx, Vy, s = 0.1)
                    ax1.set_xlabel('$V_x$ / $V$')
                    ax1.set_ylabel('$V_y$ / $V$')

                    H = self.dataDict[index]['H']
                    B = self.dataDict[index]['B']
                    ax2 = fig.add_subplot(122)
                    ax2.scatter(H,B, s = 0.1)
                    ax2.set_xlabel('H / ' + r'$Am^{-1}$')
                    ax2.set_ylabel('B / Tesla')
                
                else:
                    H = self.dataDict[index]['H']
                    B = self.dataDict[index]['B']
                    plt.scatter(H,B, s = 0.1)
                    plt.title(f'Single hysteresis loop for {self.materialName}, index: {index}')
                    plt.xlabel('H / ' + r'$Am^{-1}$')
                    plt.ylabel('B / Tesla')

                plt.tight_layout()
                if outputMode == 'save':
                    plt.savefig(f"Hysteresis-Exp Data-{self.filePath.split('.')[0]}-{index}.pdf")
                    plt.close()
                elif outputMode == 'show':
                    plt.show()

        else:
            keys = self.dataDict['combined'].keys()
            if 'Vx' in keys:
                fig = plt.figure(figsize = (10, 5))
                fig.suptitle(f'Hysteresis plot over {count} periods at 50 Hz for {self.materialName}')
                Vx = self.dataDict['combined']['Vx']
                Vy = self.dataDict['combined']['Vy']
                ax1 = fig.add_subplot(121)
                ax1.scatter(Vx, Vy, s = 0.1)
                ax1.set_xlabel('$V_x$ / $V$')
                ax1.set_ylabel('$V_y$ / $V$')

                H = self.dataDict['combined']['H']
                B = self.dataDict['combined']['B']
                ax2 = fig.add_subplot(122)
                ax2.scatter(H,B, s = 0.1)
                ax2.set_xlabel('H / ' + r'$Am^{-1}$')
                ax2.set_ylabel('B / Tesla')
            else:
                H = self.dataDict[index]['H']
                B = self.dataDict[index]['B']
                plt.scatter(H,B, s = 0.1)
                plt.title(f'Hysteresis plot over {count} periods at 50 Hz for {self.materialName}')
                plt.xlabel('H / ' + r'$Am^{-1}$')
                plt.ylabel('B / Tesla')

            plt.tight_layout()
            if outputMode == 'save':
                plt.savefig(f"Hysteresis-Exp Data-{self.filePath.split('.')[0]}-combined", dpi=600)
                plt.close()
            elif outputMode == 'show':
                plt.show()

        return


    def ccwSort(self, x, y):
        """ counter clockwise sort data, before performing shoelace algorithm

        Args:
            x (array): x data points
            y (array): y data points

        Returns:
            array: x sorted data points, y sorted data points
        """
        x0, y0 = np.mean(x), np.mean(y)
        r = np.sqrt( (x - x0 ) ** 2 + ( y - y0 ) ** 2 )

        cosRatio = ( x - x0 ) / r

        angles = np.where((y-y0) > 0, np.arccos(cosRatio), 2 * np.pi - np.arccos(cosRatio))
        mask = np.argsort(angles)
        xSorted = x[mask]
        ySorted = y[mask]
        
        return xSorted, ySorted


    def basicLoopAreaAnalysis(self):
        """ calculates area of the B-H loop area, using either shoelace or shapely algorithm

        Returns:
            mixed:  loopCount (number of loops), 
                    BHAreas (loop area array containing area for each split single hysteresis loop),
                    avgBHAreas (average area of all the single hysteresis loops), 
                    stdBHAreas (standard deviation of all the single hysteresis loops)
        """
        loopCount = len(self.dataDict.keys()) - 1

        BHAreas = np.array([])
        for index in range(loopCount):
            H = self.dataDict[index]['H']
            B = self.dataDict[index]['B']

            # method 1: shoelace
            HSorted, BSorted = self.ccwSort(H, B)
            AreaShoelace = 0.5 * np.abs ( np.dot ( HSorted, np.roll(BSorted,1)) - np.dot(BSorted,np.roll(HSorted,1 ) ) )

            # method 2: shapely
            pgon = Polygon(zip(H, B))
            AreaShapely = pgon.area
        
            if self.areaAnalysisMode == "shoelace":
                BHAreas = np.append(BHAreas, AreaShoelace)

            elif self.areaAnalysisMode == "shapely":
                BHAreas = np.append(BHAreas, AreaShapely)

        avgBHAreas = np.mean(BHAreas)
        stdBHAreas = np.std(BHAreas)
        print(self.areaAnalysisMode, self.materialName)
        return loopCount, BHAreas, avgBHAreas, stdBHAreas


    def monteCarloLoopAreaErrCal(self, count, distributionSize, pickleDumpName):
        RawVx = self.dataDict[0]['Vx']
        RawVy = self.dataDict[0]['Vy']

        """
        RpRange = np.random.normal(loc=self.Rp,     scale=0,     size=distributionSize)
        LpRange = np.random.normal(loc=self.Lp,     scale=0,     size=distributionSize)
        CRange = np.random.normal(loc=self.C,       scale=0,     size=distributionSize)
        RsRange = np.random.normal(loc=self.Rs,     scale=0,     size=distributionSize)
        AsRange = np.random.normal(loc=self.As,     scale=0,     size=distributionSize)
        """
        print(f'Rp: {self.Rp}   {self.RpError}')
        print(f'Lp: {self.Lp}   {self.LpError}')
        print(f'C: {self.C}   {self.CError}')
        print(f'Rs: {self.Rs}   {self.RsError}')
        print(f'As: {self.As}   {self.AsError}')
        
        RpRange = np.random.normal(loc=self.Rp,     scale=self.RpError,     size=distributionSize)
        LpRange = np.random.normal(loc=self.Lp,     scale=self.LpError,     size=distributionSize)
        CRange = np.random.normal(loc=self.C,       scale=self.CError,      size=distributionSize)
        RsRange = np.random.normal(loc=self.Rs,     scale=self.RsError,     size=distributionSize)
        AsRange = np.random.normal(loc=self.As,     scale=self.AsError,     size=distributionSize)
        
        AreaShapelyList = []
        for i in range(count):
            # loop construction
            HArray = []
            for voltageX in RawVx:
                H = ( self.Np * voltageX ) / ( random.choice(LpRange) * random.choice(RpRange) )
                HArray.append(H)

            BArray = []
            for voltageY in RawVy:
                B = ( random.choice(RsRange) * random.choice(CRange) * voltageY ) / ( self.Ns * random.choice(AsRange))
                BArray.append(B)

            # loop area calculation
            pgon = Polygon(zip(HArray, BArray))
            AreaShapely = pgon.area
            AreaShapelyList.append(AreaShapely)
            
            print(i, np.mean(AreaShapelyList), np.std(AreaShapelyList))

        self.pickleDump(target=AreaShapelyList, name=pickleDumpName)

        return AreaShapelyList


    def monteCarloLoopAreaErrAnal(self, pickleFilePath, plot, plotInterval, countStart, countEnd, deltaPlotInterval, deltaCountStart, deltaCountEnd):
        data = np.array(self.pickleOpen(name=pickleFilePath))[countStart:countEnd]
        countList = range(len(data))

        meanList, STDList = [], []
        meanPrevDeltaList, STDPrevDeltaList = [], []
        for i in countList:
            mean = np.mean(data[:i+1])
            std = np.std(data[:i+1])

            meanList.append(mean)
            STDList.append(std)
            if i != 0:
                meanPrevDeltaList.append(((meanList[-1] - meanList[-2]) * 100 ) / meanList[-2])
                STDPrevDeltaList.append(((STDList[-1] - STDList[-2]) * 100) / STDList[-2])

        plotCountList = countList[countStart:countEnd:plotInterval]
        plotMeanList = meanList[countStart:countEnd:plotInterval]
        plotSTDList = STDList[countStart:countEnd:plotInterval]

        plotPrevDeltaCountList = countList[deltaCountStart:deltaCountEnd:deltaPlotInterval]
        plotmeanPrevDeltaList = meanPrevDeltaList[deltaCountStart:deltaCountEnd:deltaPlotInterval]
        plotSTDPrevDeltaList = STDPrevDeltaList[deltaCountStart:deltaCountEnd:deltaPlotInterval]

        """
        space = 100
        plotMovingAverge, plotMovingAvergeDelta = [], []
        for i in range(int(len(data) / space) - 2):
            plotMovingAverge.append(np.mean(meanList[i*100:i*100+100]))
            if len(plotMovingAverge) > 1:
                plotMovingAvergeDelta.append(np.abs(plotMovingAverge[-1]-plotMovingAverge[-2]))

        plt.plot(range(len(plotMovingAvergeDelta)), plotMovingAvergeDelta)
        plt.show()
        """

        if plot:
            fig = plt.figure(figsize = (12, 6))
            fig.suptitle(f'Monte Carlo Error Analysis of Single Hysteresis Loop Area', fontsize=14)
            
            ax1 = fig.add_subplot(121)

            ax1.set_xlabel("Iterations", color="black", fontsize = 12)

            ax1.plot(plotCountList, plotMeanList, color="blue", linestyle="-", linewidth = 0.5)
            ax1.set_ylabel("Cumulative mean", color='blue', fontsize = 12)
            ax1.set_yticks(np.arange(13000, 15100, 500))
            ax1.set_xticks(np.arange(0, 85000, 20000))
            
            ax2 = ax1.twinx()
            ax2.plot(plotCountList, plotSTDList, color="red", linestyle="-", linewidth = 0.5)
            ax2.set_ylabel("Cumulative STD",color="red",fontsize = 12)
            ax2.set_yticks(np.arange(7000, 8100, 250))
            

            ax3 = fig.add_subplot(122)
            ax3.plot(plotPrevDeltaCountList, plotmeanPrevDeltaList, color="blue", linestyle="-", linewidth = 0.5, label = 'Mean')
            ax3.plot(plotPrevDeltaCountList, plotSTDPrevDeltaList, color="red", linestyle="-", linewidth = 0.5, label = 'STD')
            ax3.set_xlabel("Iterations", color="black", fontsize = 12)
            ax3.set_ylabel("Fluctuations in Mean and STD / %", color='black', fontsize = 12)
            ax3.legend(loc = "lower right")
            ax3.set_xticks(np.arange(0, 2001, 500))

            ax4 = plt.axes([0.75, 0.6, .2, .2], facecolor='w')
            plt.plot(plotPrevDeltaCountList[-100:], plotmeanPrevDeltaList[-100:], color="blue", linestyle="-", linewidth = 0.5)
            plt.plot(plotPrevDeltaCountList[-100:], plotSTDPrevDeltaList[-100:], color="red", linestyle="-", linewidth = 0.5)
            #plt.title('Impulse response')
            #plt.xlim(0, 0.2)
            plt.xticks([1800, 1900, 2000])
            plt.yticks([-0.2, 0, 0.2])

            plt.tight_layout(pad=2.0)
            plt.show()
        return meanList, STDList


    def gradientCorrection(self, index, gradientH):
        # index decides which loop to use, single loop analysis, does not really matter

        rawH = self.dataDict[index]['H']
        rawB = self.dataDict[index]['B']
        datapointCount = len(rawH)

        print(f"Data points in raw data: {datapointCount}")

        # first we remove all adjacent Hs that have the same value, to avoid generating singularities in the gradient
        zeroCleanH = []
        zeroCleanB = []

        for i in range(datapointCount - 1):
            deltaH = rawH[i+1] - rawH[i]
            if deltaH != 0:
                zeroCleanH.append(rawH[i])
                zeroCleanB.append(rawB[i])

        """
        plt.scatter(zeroCleanH, zeroCleanB, s = 0.5)
        plt.title(r'Hysteresis curve $B$ vs $H$')
        plt.xlabel(r'$H/Am^{-1}$')
        plt.ylabel(r'$B/T$')
        plt.show()
        """

        # perform linear regression using H values above the defined gradientH

        linearH = [i[0] for i in list(filter(lambda x: (x[0] > gradientH), zip(zeroCleanH, zeroCleanB)))  ]
        linearB = [i[1] for i in list(filter(lambda x: (x[0] > gradientH), zip(zeroCleanH ,zeroCleanB)))  ]
        residualGradient = stats.linregress(linearH, linearB)
        print(f"residual gradient: {residualGradient.slope}")

        # adjust gradient using resdiual gradient for all the data points
        zeroZippedBH = list(zip(zeroCleanH,zeroCleanB))
        adjustedH = []
        adjustedB = []
        adjustedH.append(zeroZippedBH[0][0])
        adjustedB.append(zeroZippedBH[0][1])

        for i in range(len(zeroZippedBH) - 1):
            presentH, nextH = zeroZippedBH[i][0], zeroZippedBH[i+1][0]
            presentB, nextB = zeroZippedBH[i][1], zeroZippedBH[i+1][1]
            deltaH = nextH - presentH
            deltaB = nextB - presentB
            gradient = deltaB / deltaH
            adjustedH.append(nextH)
            lastB = adjustedB[-1]
            adjustedB.append(lastB + (gradient - residualGradient.slope) * deltaH)

        """
        plt.scatter(adjustedH,adjustedB, s = 0.5)
        plt.title(r'Adjusted hysteresis curve $B$ vs $H$')
        plt.xlabel(r'$H/Am^{-1}$')
        plt.ylabel(r'$B/T$')
        plt.show()
        """

        self.adjustedH, self.adjustedB = adjustedH, adjustedB

        return adjustedH, adjustedB


    def fitJAModel(self, truncateHList, spliceInterval, iniMagDeltaH, bnds, iniParamsList, tol, fitMethod):
        
        # storing it right away for later use
        self.iniMagDeltaH = iniMagDeltaH

        # first truncate terms above and below a certain H, which requires some trial and error so we put in some values for truncateHList
        JAModelDict = {}
        for cutoffTerm in truncateHList:
            
            # cutting
            print(f"truncate H above and below:  {cutoffTerm}")
            truncatedH = [i[0] for i in list(filter(lambda x: (x[0] < cutoffTerm and x[0] > -cutoffTerm), zip(self.adjustedH, self.adjustedB)))  ]
            truncatedB = [i[1] for i in list(filter(lambda x: (x[0] < cutoffTerm and x[0] > -cutoffTerm), zip(self.adjustedH, self.adjustedB)))  ]

            # reducing datapoints if needed
            sparseH = truncatedH[::spliceInterval]
            sparseM = [tuple[1] / (4e-7 * np.pi ) for tuple in zip(truncatedH, truncatedB)][::spliceInterval]

            # Julia model starts generating data points at the uppermost and rightmost point in the loop, i.e., largest M and H values
            # we reorder sparseH and sparseM arrays to match that for a proper comparison

            maxHPos = 0
            while sparseH[maxHPos+1] > sparseH[maxHPos]:
                maxHPos+=1

            for i in range(maxHPos):
                sparseH.append(sparseH[0])
                sparseH.pop(0)
                sparseM.append(sparseM[0])
                sparseM.pop(0)

            self.sparseH = sparseH
            self.sparseM = sparseM

            """
            plt.scatter(sparseH, sparseM, s = 0.5)
            plt.title(r'Adjusted hysteresis curve $M$ vs $H$')
            plt.xlabel(r'$H$')
            plt.ylabel(r'$M$')
            plt.show()
            """

            # Fitting JA model
            self.sparseDatapointCount = len(self.sparseH)
            print(f"sparseDatapointCount: {self.sparseDatapointCount}")


            """
            for i in range(len(sparseH)):
                print(sparseH[i], sparseM[i])
            """

            parameters = minimize(self.objectiveFuncJAModel, iniParamsList, tol=tol, method = fitMethod, bounds=bnds)
            print(parameters)
            print(f"cutoffTerm: {cutoffTerm}")

            sparseDict = {}
            sparseDict['sparseH'] = sparseH
            sparseDict['sparseM'] = sparseM
            sparseDict['minimizeParameters'] = parameters
            sparseDict['iniMagDeltaH'] = iniMagDeltaH
            
            JAModelDict[cutoffTerm] = sparseDict

        with open("JAModelDict", 'wb') as f:
            pickle.dump(JAModelDict, f)
            """
            plt.scatter(H[Nfirst:], M[Nfirst:], s=1)
            plt.show()
            """


    def JAModelGenerator(self, paramsList, Nfirst):
        
        # set up JA model
        H = [0]
        delta = [0]
        Man = [0]
        dMirrdH = [0]
        Mirr = [0]
        M = [0]

        for i in range(Nfirst):
            H.append(H[i] + self.iniMagDeltaH)

        for i in self.sparseH:
            H.append(i)

        for i in range(len(H) - 1):
            if H[i + 1] > H[i]:
                delta.append(1)
            else:
                delta.append(-1)

        a = paramsList[0]
        alpha = paramsList[1]
        c = paramsList[2]
        k = paramsList[3]
        Ms = paramsList[4]
        
        # Ms = 4.8e4

        for i in range(len(H) - 1):
            Man.append(Ms * (1 / np.tanh((H[i + 1] + alpha * M[i]) / a) - a / (H[i + 1] + alpha * M[i])))
            dMirrdH.append((Man[i+1] - M[i]) / (k * delta[i+1] - alpha * (Man[i + 1] - M[i])))
            Mirr.append(Mirr[i] + dMirrdH[i + 1] * (H[i+1] - H[i]))
            M.append(c * Man[i + 1] + (1 - c) * Mirr[i + 1])
        
        return H, M


    def objectiveFuncJAModel(self, paramsList):

        Nfirst = int(self.sparseH[0] / self.iniMagDeltaH)

        JAmodelH, JAmodelM = self.JAModelGenerator(paramsList=paramsList, Nfirst=Nfirst)
        
        truncJAmodelH = JAmodelH[Nfirst+1:]
        truncJAmodelM = JAmodelM[Nfirst+1:]

        residualSquared = 0

        for i in range(len(truncJAmodelH) - 1):
            residualSquared += (self.sparseM[i] - truncJAmodelM[i]) ** 2
        
        scaledDifference = residualSquared / self.sparseDatapointCount
        print(f"scaledDifference: {scaledDifference}")
        # print(residualSquared)
        return scaledDifference


    def compareJAExpData(self, truncateHList):
        with open("JAModelDict", 'rb') as f:
            JAModelDict = pickle.load(f)

        for key in truncateHList:
            if key in JAModelDict.keys():
                sparseM = JAModelDict[key]["sparseM"]
                sparseH = JAModelDict[key]["sparseH"]
                
                paramsList = JAModelDict[key]['minimizeParameters'].x
                fun = JAModelDict[key]['minimizeParameters'].fun
                success = JAModelDict[key]['minimizeParameters'].success

                print(f"key: {key}")
                print(f"sparseH[0]: {sparseH[0]}")
                print(f"fun: {fun}")
                print(f"str(paramsList): {str(paramsList)}")
                print(f"str(success): {str(success)}")

                iniMagDeltaH = JAModelDict[key]['iniMagDeltaH']
                Nfirst = int(sparseH[0] / iniMagDeltaH)

                self.sparseH = sparseH
                self.iniMagDeltaH = iniMagDeltaH

                JAModelH, JAModelM = self.JAModelGenerator(paramsList, Nfirst=Nfirst)
                
                truncJAModelH = JAModelH[Nfirst+1:]
                truncJAModelM = JAModelM[Nfirst+1:]
                
                datapointCount = len(truncJAModelH)
                plt.title(f"JA model best fit with truncation above H = {key}", pad=20)
                plt.xlabel(f"H")
                plt.xlabel(f"")
                plt.scatter(truncJAModelH, truncJAModelM, s=0.2, label = 'JA model')
                plt.scatter(sparseH, sparseM, s=0.2, label = 'exp. data')
                plt.xlabel('H / ' + r'$Am^{-1}$')
                plt.ylabel('M / ' + r'$Am^{-1}$')
                lgnd = plt.legend()
                lgnd.legendHandles[0]._sizes = [20]
                lgnd.legendHandles[1]._sizes = [20]


                plt.show()

                # chi squared cal.
                chiSquared = 0
                for i in range(len(truncJAModelM)):
                    # chiSquared +=  ( ( truncJAModelM[i] - sparseM[i] ) / ( sparseM[i] * self.BFracError ) ) ** 2

                    chiSquared += ( ( truncJAModelM[i] - sparseM[i] ) / ( np.abs(truncJAModelM[i]) ** 0.5 ) ) ** 2

                print(f"chiSquared: {int(chiSquared)}")
                print(f"datapointCount: {datapointCount}")

                # branch deviation plot for residuals
                truncJAModelHOne = np.array(truncJAModelH[:int(datapointCount/2)])
                truncJAModelHTwo = np.array(truncJAModelH[int(datapointCount/2):])
                
                truncJAModelMOne = np.array(truncJAModelM[:int(datapointCount/2)])
                truncJAModelMTwo = np.array(truncJAModelM[int(datapointCount/2):])

                sparseMOne = np.array(sparseM[:int(datapointCount/2)])
                sparseMTwo = np.array(sparseM[int(datapointCount/2):])

                branchOneResidual = [(sparseMOne[i] - truncJAModelMOne[i]) / truncJAModelMOne[i] for i in range(len(truncJAModelHOne))]
                branchTwoResidual = [(sparseMTwo[i] - truncJAModelMTwo[i]) / truncJAModelMTwo[i] for i in range(len(truncJAModelHTwo))]
                
                """
                for i in range(len(truncJAModelHOne)):
                    if i % 10 == 0:
                        print(truncJAModelHOne[i], truncJAModelMOne[i], sparseMOne[i])

                print("SPACER")
                for i in range(len(truncJAModelHTwo)):
                    if i % 10 == 0:
                        print(truncJAModelHTwo[i], truncJAModelMTwo[i], sparseMTwo[i])
                """
                branchOneResidual = sparseMOne - truncJAModelMOne
                branchTwoResidual = sparseMTwo - truncJAModelMTwo

                plt.plot(truncJAModelHOne, branchOneResidual, color="blue", linestyle="-", linewidth = 0.5, label="upper branch")
                plt.plot(truncJAModelHTwo, branchTwoResidual, color="red", linestyle="-", linewidth = 0.5, label="lower branch")
                plt.title(f"Residuals of the JA model with truncation above H = {key}", pad=20)
                plt.xlabel('H / ' + r'$Am^{-1}$')
                plt.ylabel('M / ' + r'$Am^{-1}$')
                plt.legend()
                plt.show()

        return