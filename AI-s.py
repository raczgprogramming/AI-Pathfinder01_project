class AI:
    def __init__(self,name,alpha,gamma,epsilon,epsilon_decay,epsilon_min):
        self.name=name
        self.alpha=alpha
        self.gamma=gamma
        self.epsilon=epsilon
        self.epsilon_decay=epsilon_decay
        self.epsilon_min=epsilon_min


al = AI("Al",0.1,0.9,0.1,0,0.1)

fred = AI("Fred",0.1,0.9,1.0,0.995,0.01)