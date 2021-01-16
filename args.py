

class QArgs():
    def __init__(self):
        self.gamma = 0.95
        self.eta = 0.1
        self.num_of_projects_to_start = 10
        self.num_of_iters = 100
        self.iter_to_choose_best_action = 0
        self.num_episodes = 100
        self.threshold = -0.5
        self.epsilon = 0.1

    def getEta(self, iter):
        if(iter < self.num_of_iters/4):
            return 0.9
        elif(iter < self.num_of_iters/2):
            return 0.5
        elif(iter < 3*self.num_of_iters/4):
            return 0.3
        else:
            return 0.1

    def getEpsilon(self, iter):
        if(iter < self.num_of_iters/4):
            return 0.1
        elif(iter < self.num_of_iters/2):
            return 0.05
        elif(iter < self.num_of_iters/5):
            return 0.02
        else:
            return 0.001
