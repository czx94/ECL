def score(p1, p2):
    # res = (p1.pPair*p2.pPair) + ((1-p1.pPair)*(1-p2.pPair)) - (p1.pPair*(1-p2.pPair)) - ((1-p1.pPair)*p2.pPair)
    res = (p1 - (1 - p1)) * (p2- (1 - p2))
    # print(res)
    return res, -res

print(score(0.5540128407, 0.4634499945))