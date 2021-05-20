class Config:
    def __init__(self,
                 device="cpu",
                 learning_rate=0.001,
                 learning_rate_decay=1,
                 epsilon = 0.5,
                 epsilon_low = 0.1,
                 epsilon_step = 0.05,
                 epochs=1000,
                 max_play_length=50,
                 gamma = 0.99
                 ):

        self.device=device
        self.learning_rate=learning_rate
        self.learning_rate_decay=learning_rate_decay
        self.epsilon = epsilon
        self.epsilon_low = epsilon_low
        self.epsilon_step = epsilon_step
        self.epochs=epochs
        self.max_play_length=max_play_length
        self.gamma = gamma
