import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.stats import bernoulli
import random

# ####################################################### probleme0##################################################
#
# # q1
# # x=np.linspace(-1,1,10000)
# # y=-(x**2*(2+np.sin(x))**2)
# # plt.plot(x,y)
# # plt.show()
# # print('Max:',max(y))
#
# def fit(x):
#     return abs(x**2*(2+np.sin(x))**2)
#
# #q2
# class Individual():
#     def __init__(self):
#         self.v = np.random.uniform(-1,1)
#         self.fitness = fit(self.v)
#
# class GA():
#     def __init__(self,pop, pgm, ppm, ps):
#         self.popu = [Individual() for i in range(pop)]
#         self.N = pop
#         self.pgm = pgm
#         self.ppm = ppm
#         self.ps = ps
#         self.newGene = []
#         self.best = self.popu[0]
#
#     def mutation(self):
#         for i in range(int(self.N*self.pgm)):
#             self.newGene.append(Individual())
#
#     def pMutation(self, inds, epsilon):
#         for ind in inds[:int(self.ppm*self.N)]:
#             ind = ind[1]
#             ind.v += np.random.uniform(-epsilon, epsilon)
#             ind.fitness = fit(ind.v)
#             self.newGene.append(ind)
#
#     def crossover(self):
#         for i in range(int(self.N*self.ps)):
#             p1, p2 = self.select(self.popu, 2)
#             child = Individual()
#             child.v = (p1.v + p2.v)/2
#             child.fitness = fit(child.v)
#
#             self.newGene.append(child)
#
#     def select(self,inds, n):
#         return np.random.choice(inds, n)
#
#     def cycle(self):
#         dic = {p.fitness: p for p in self.popu}
#         res = sorted(dic.items(), key=lambda dic: dic[0])
#         self.best = res[0][1]
#         self.em = self.best.v / 5
#
#         #training
#         self.mutation()
#         self.pMutation(res, self.em)
#         self.crossover()
#
#         keep = self.N - len(self.newGene)
#
#         for val, ind in res:
#             temp = []
#             temp.append(ind)
#             for i in temp[:keep-1]:
#                 self.newGene.append(i)
#
#         self.popu = self.newGene
#         self.newGene = []
#
#     def train(self, prec):
#         t1 = time.time()
#         self.cycle()
#         count = 1
#         while self.best.fitness > prec:
#             count += 1
#             self.cycle()
#         t2 = time.time()
#         print('epoch:%d, precision:%.10f,error:%.15f,size:%d, time:%f'%(count, prec, self.best.fitness, self.N, t2 - t1))
#
# #b1
# precs = [10**(-m) for m in range(7,9)]
# print('B1')
# for p in precs:
#     ga1 = GA(10, 0.8, 0, 0)
#     ga1.train(p)
#     del ga1
#
# #b2
# print('B2')
# for p in precs:
#     ga2 = GA(10, 0.8, 0.1, 0)
#     ga2.train(p)
#     del ga2
#
# #b3
# print('B3')
# size = [10, 100, 1000, 10000, 100000]
# for s in size:
#     ga3 = GA(s, int((s-1)/3),int((s-1)/3),int((s-1)/3))
#     ga3.train(10E-8)

####################################################### probleme1##################################################
#q1
#Le gain pour joueur1 E1 = p11 * p21 + p21 * p22 - p11 * p22 - p21 * p12

class Player():
    def __init__(self):
        self.pPair = random.random()
        self.score = 0

class Game():
    def __init__(self, size, eps):
        self.player1 = [Player() for i in range(size)]
        self.player2 = [Player() for i in range(int(size))]
        self.eps = eps
        self.size = size

    def mutation(self,p1,p2):
        p2.pPair = np.random.uniform(max(p1.pPair - self.eps, 0), min(p1.pPair + self.eps, 1))
        print(p1.pPair,p2.pPair)

    # def pMutation(self, inds, epsilon, newgene):
    #     for ind in inds[:int(self.ppm*self.size)]:
    #         ind = ind[1]
    #         ind.pPair += np.random.uniform(-epsilon, epsilon)
    #         newgene.append(ind)

    # def crossover(self):
    #     for i in range(int(self.size*self.ps)):
    #         p1, p2 = self.select(self.player1, 2)
    #         child = Player()
    #         child.pPair = (p1.pPair + p2.pPair)/2
    #         child.score = (p1.score + p2.score)
    #
    #         self.newG1.append(child)
    #
    #     for i in range(int(self.size*self.ps)):
    #         p1, p2 = self.select(self.player2, 2)
    #         child = Player()
    #         child.pPair = (p1.pPair + p2.pPair)/2
    #         child.score = (p1.score + p2.score)
    #
    #         self.newG2.append(child)
    #
    # def select(self,inds, n):
    #     return np.random.choice(inds, n)

    def play(self, game_number):
        for p1 in self.player1:
            for p2 in self.player2:
                for i in range(game_number):
                    e1, e2 = self.score(p1, p2)
                    p1.score += e1
                    p2.score += e2

    def score(self, p1, p2):
        #res = (p1.pPair*p2.pPair) + ((1-p1.pPair)*(1-p2.pPair)) - (p1.pPair*(1-p2.pPair)) - ((1-p1.pPair)*p2.pPair)
        res = (p1.pPair-(1-p1.pPair))*(p2.pPair-(1-p2.pPair))
        #print(res)
        return res, -res

    def generate(self, game_number):
        self.play(game_number)
        dic1 = {p.score: p for p in self.player1}
        dic2 = {p.score: p for p in self.player2}

        res1 = sorted(dic1.items(), key=lambda dic: dic[0])[::-1]
        res2 = sorted(dic2.items(), key=lambda dic: dic[0])[::-1]
        m1 = np.mean([r[0] for r in res1])
        m2 = np.mean([r[0] for r in res2])

        best1, worst1 = res1[0][1], res1[-1][1]
        best2, worst2 = res2[0][1], res2[-1][1]

        self.mutation(best1,worst1)
        self.mutation(best2,worst2)
        # self.pMutation(res1, self.eps, self.newG1)
        # self.pMutation(res2, self.eps, self.newG2)
        # self.crossover()
        print(best1.pPair, best2.pPair, self.score(best1, best2))

        return self.score(best1, best2)[0], best1, best2

    def train(self, game_number, iter):
        t1 = time.time()
        pl = []
        for i in range(iter):
            print(i)
            # if i % 1000 == 0:
            #     self.eps /= 2
            #self.eps = self.eps * np.exp(-iter)
            a, p1, p2 = self.generate(game_number)
            pl.append(a)
        t2 = time.time()
        print('P1 score:%d, P1(pair):%.10f, P2 score:%f, P2(pair):%.10f, time:%f'%(p1.score, p1.pPair,p2.score, p2.pPair, t2 - t1))
        temps = list(range(iter))
        plt.plot(temps, pl, 'blue')
        plt.show()


game = Game(10, 0.03)
game.train(1, 10000)
print()
############################################################problem2#####################################################################3
# class Player():
#     def __init__(self):
#         self.p1 = self.geProba()
#         self.score = 0
#
#     def proba(self,v):
#         return self.p1[int(10*v)]
#
#     def geProba(self):
#         proba = [random.random() for i in range(10)]
#         proba = [i/sum(proba) for i in proba]
#         return proba
#
# class Game():
#     def __init__(self, size, eps):
#         self.player1 = [Player() for i in range(size)]
#         self.player2 = [Player() for i in range(int(size))]
#         self.eps = eps
#         self.size = size
#
#     def mutation(self,p1,p2):
#         a = int(np.random.uniform(0, 10))
#         p2.p1[a] = np.random.uniform(max(p1.p1[a] - self.eps, 0), min(p1.p1[a] + self.eps, 1))
#         s = sum(p2.p1)
#         p2.p1 =[i/s for i in p2.p1]
#
#     def play(self, game_number):
#         for p1 in self.player1:
#             for p2 in self.player2:
#                 for i in range(game_number):
#                     e1, e2 = self.score(p1, p2)
#                     p1.score += e1
#                     p2.score += e2
#
#     def pari(self, a1, a2):
#         if a1 == a2:
#             return 0
#         else:
#             win = (1 + a1) if (a1 > a2) else -(1 + a1)
#         #print(a1,a2,win)
#         return win
#
#     def score(self, p1, p2):
#         a1 = random.random()
#         a2 = random.random()
#         res = p1.proba(a1) * (1-p2.proba(a2)) + p1.proba(a1)*p2.proba(a2)*self.pari(a1, a2) - (1-p1.proba(a1))*p2.proba(a2)
#         return res, -res
#
#     def crossover(self, p1, p2):
#         a = int(np.random.uniform(0, 10))
#         p1.p1[a] = (p1.p1[a] + p2.p1[a]) / 2
#         p2.p1[a] = (p1.p1[a] + p2.p1[a]) / 2
#         s1 = sum(p1.p1)
#         s2 = sum(p2.p1)
#         p1.p1 = [i / s1 for i in p2.p1]
#         p2.p1 = [i / s2 for i in p2.p1]
#
#     def generate(self, game_number):
#         self.play(game_number)
#         dic1 = {p.score: p for p in self.player1}
#         dic2 = {p.score: p for p in self.player2}
#
#         res1 = sorted(dic1.items(), key=lambda dic: dic[0])[::-1]
#         res2 = sorted(dic2.items(), key=lambda dic: dic[0])[::-1]
#         m1 = np.mean([r[0] for r in res1])
#         m2 = np.mean([r[0] for r in res2])
#         # print(m1,m2)
#         best1, worst1 = res1[0][1], res1[-1][1]
#         best2, worst2 = res2[0][1], res2[-1][1]
#
#         self.mutation(best1,worst1)
#         self.mutation(best2,worst2)
#         self.crossover(res1[0][1], res1[1][1])
#         self.crossover(res2[0][1], res2[1][1])
#
#         return m1, m2, best1, best2
#
#     def train(self, game_number, iter):
#         t1 = time.time()
#         pl = []
#         pll = []
#         for i in range(iter):
#             print(i)
#             #if i % 1000 == 0:
#                 #self.eps /= 2
#             #self.eps = self.eps * np.exp(-iter)
#             a,b, p1, p2 = self.generate(game_number)
#             pl.append(a)
#             pll.append(b)
#         t2 = time.time()
#         print(p1.score, p1.p1)
#         print(p2.score, p2.p1)
#         temps = list(range(iter))
#         plt.plot(temps, pl, 'blue', temps, pll, 'red')
#         plt.show()
#
# game = Game(10, 0.05)
# game.train(1, 30000)