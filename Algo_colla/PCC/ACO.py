import numpy as np
import pandas as pd
from math import pow, sqrt
from tqdm import tqdm
import pylab as pl

def sigmoid(x):
    return 1/(1+np.exp(-x))

class TSP():
    def __init__(self, city_file, antNumber = 20, alpha = 7, rou = 0.3, beta = 7, q = 20, iterMax = 10000):
        self.alpha = alpha
        self.beta = beta
        self.q = q
        self.rou = rou
        self.iterMax = iterMax
        self.antNumber = antNumber
        self.antList = []
        self.cities = self.ReadCityInfo(city_file)
        self.Shortest = float('inf')

    def ReadCityInfo(self, fileName):
        city_info = pd.read_csv(fileName,
                                sep=' ',
                                skiprows=6, skipfooter=1,
                                engine='python',
                                header=None,
                                names=('N', 'x', 'y'))
        self.CityCount = city_info.shape[0]
        self.CitySet = set()
        self.CityDistance = [[0] * self.CityCount for i in range(self.CityCount)]
        self.Pheromone = [[np.random.normal(10,1,1)] * self.CityCount for i in range(self.CityCount)]
        self.PheromoneDelta = np.zeros((self.CityCount, self.CityCount))
        self.BestTour = [None] * self.CityCount

        for row in city_info.index:
            for col in city_info.index:
                if row != col:
                    distance = round(sqrt(pow(city_info.x[row] - city_info.x[col], 2) + pow(city_info.y[row] - city_info.y[col], 2)))
                    self.CityDistance[row][col] = distance
        return city_info

    def antInit(self):
        self.antList.clear()
        for i in range(self.antNumber):
            self.antList.append(Ant(self.alpha, self.beta, np.random.choice(self.cities.index), self.cities.index, self.CityDistance, self.Pheromone))

    def search(self):
        for iter in tqdm(range(self.iterMax)):
            self.antInit()
            tmpLen = float('inf')
            tmpTour = []
            for ant in self.antList:
                for ttt in range(self.CityCount):
                    ant.movetoNext()
                ant.two_opt_search()
                ant.UpdatePathLen()
                if ant.CurrLen < tmpLen:
                    self.bestAnt = ant
                    tmpLen = ant.CurrLen
                    tmpTour = ant.TabuCityList
            if tmpLen < self.Shortest:
                self.Shortest = tmpLen
                self.BestTour = tmpTour
            print(self.Shortest, ":", self.BestTour)
            self.UpdatePheromoneTrail()

            pl.clf()
            x = []
            y = []
            for city in self.BestTour:
                x.append(self.cities.x[city])
                y.append(self.cities.y[city])
            x.append(x[0])
            y.append(y[0])
            pl.plot(x, y)
            pl.scatter(x, y, s=30, c='r')
            pl.pause(0.01)

    def UpdatePheromoneTrail(self):
        ant = self.bestAnt
        # best_current = self.bestAnt.CurrLen
        # for ant in self.antList:
        pheromo_new = self.q  / ant.CurrLen
        tabu = ant.TabuCityList
        PD = self.PheromoneDelta
        P = self.Pheromone
        citys = self.cities.index

        for city, nextCity in zip(tabu[:-1], tabu[1:]):
            PD[city][nextCity] = pheromo_new
            PD[nextCity][city] = pheromo_new
        lastCity = tabu[-1]
        firstCity = tabu[0]
        PD[lastCity][firstCity] = pheromo_new
        PD[firstCity][lastCity] = pheromo_new

        for c1 in citys:
            for c2 in citys:
                if c1 != c2:
                    P[c1][c2] = (
                        (1 - self.rou) * P[c1][c2]
                        + PD[c1][c2]
                    )
                    if P[c1][c2] < 0.001:
                        P[c1][c2] = 0.001
                    if P[c1][c2] > 10:
                        P[c1][c2] = 10
                    PD[c1][c2] = 0
        # #做归一化之后好像经常不能收敛到全局最优
        # for c1 in citys:
        #     for c2 in citys:
        #         if c1 != c2:
        #             P[c1][c2] = ((1 - self.rou) * P[c1][c2] + PD[c1][c2])
        #             PD[c1][c2] = 0
        #
        # maxi = np.amax(P)
        # mini = np.amin(P)
        # for c1 in citys:
        #     for c2 in citys:
        #         if c1 != c2:
        #             P[c1][c2] = 10*(P[c1][c2] - mini+0.1)/(maxi-mini)

class Ant():
    def __init__(self, alpha, beta, currCity = 0, cities = None, cityDis = None, pheromo=None):
        self.TabuCityList = [currCity, ]
        self.AllowedCitySet = set(cities)
        self.AllowedCitySet.remove(currCity)
        self.CityDistance = cityDis
        self.Pheromone = pheromo
        self.TransferProbabilityList = []
        self.CurrCity = currCity
        self.CurrLen = 0
        self.sensitive = sigmoid(np.random.standard_normal(1))
        self.alpha = alpha * self.sensitive
        self.beta = beta * (1 - self.sensitive)

    def selectCity(self):
        sumProba = 0
        sumProbaList = []
        for city in self.AllowedCitySet:
            sumProba += (pow(self.Pheromone[self.CurrCity][city], self.alpha) * pow(1.0 / self.CityDistance[self.CurrCity][city], self.beta))
            temp = sumProba
            sumProbaList.append((city, temp))
        threshold = sumProba*np.random.random()
        for cityNum, cityProb in sumProbaList:
            if threshold <= cityProb:
                return cityNum

    def movetoNext(self):
        nextCity = self.selectCity()
        if nextCity is not None:
            self.CurrCity = nextCity
            self.TabuCityList.append(nextCity)
            self.AllowedCitySet.remove(nextCity)

    def UpdatePathLen(self):
        for city, nextCity in zip(self.TabuCityList[:-1],self.TabuCityList[1:]):
            self.CurrLen += self.CityDistance[city][nextCity]
        lastCity = self.TabuCityList[-1]
        firstCity = self.TabuCityList[0]
        self.CurrLen += self.CityDistance[lastCity][firstCity]

    def two_opt_search(self):
        '''
        1-2-3-4, 1-2 + 3-4 > 1-3 + 2-4 则交换
        '''
        cityNum = len(self.TabuCityList)
        for i in range(cityNum):
            for j in range(cityNum - 1, i, -1):
                curCity1 = self.TabuCityList[i]
                preCity1 = self.TabuCityList[(i - 1) % cityNum]
                curCity2 = self.TabuCityList[j]
                nextCity2 = self.TabuCityList[(j + 1) % cityNum]
                CurrLen = self.CityDistance[preCity1][curCity1] + self.CityDistance[curCity2][nextCity2]
                NextLen = self.CityDistance[preCity1][curCity2] + self.CityDistance[curCity1][nextCity2]
                if NextLen < CurrLen:
                    tempList = self.TabuCityList[i:j + 1]
                    self.TabuCityList[i:j + 1] = tempList[::-1]

if __name__ == '__main__':
    tsp = TSP('eil101.tsp')
    tsp.search()