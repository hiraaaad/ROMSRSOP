import collections

class PheromoneDict():
    def __init__(self, **kwargs):
        self.P = collections.defaultdict(self.getCurrentTau)
        # self.P['currentTau'] = self.getTau0()

        super().__init__(**kwargs)

    def getPheromoneValue(self, component):
        return self.P[component]

    def setPheromoneValue(self, component, value):
        self.P[component] = value

    def sharePheromoneStructure(self, other):
        other.P = self.P

    # def getCurrentTau(self):
    #     return self.P['currentTau']

    def pheromoneDecay(self):
        for component in self.P:
            self.P[component] *= (1-self.getRho())

    def getComponents(self):
        return (c for c in self.P)