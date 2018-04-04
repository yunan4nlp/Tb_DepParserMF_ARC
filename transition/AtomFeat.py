class AtomFeat:
    def __init__(self):
        self.s0 = -1
        self.s1 = -1
        self.s2 = -1
        self.q0 = -1

        self.arc = False

    def index(self):
        return self.s0, self.s1, self.s2, self.q0, self.arc
