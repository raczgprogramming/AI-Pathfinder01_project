class AI:
    def __init__(self,name,alpha,gamma,epsilon,epsilon_decay,epsilon_min):
        self.name=name
        self.alpha=alpha
        self.gamma=gamma
        self.epsilon=epsilon
        self.epsilon_decay=epsilon_decay
        self.epsilon_min=epsilon_min