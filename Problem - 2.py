import numpy as np
from scipy.stats import poisson


class Hospital():

    def __init__(self, availableBeds, initialBedsNormal, initialBedsCovid, N, discountRate) -> None:
        self.availableBeds = availableBeds
        self.initialBedsNormal = initialBedsNormal
        self.initialBedsCovid = initialBedsCovid
        self.N = N
        self.discountRate = discountRate


class poisson_:

    def __init__(self, λ):
        self.λ = λ

        ε = 0.01

        # [α , β] is the range of n's for which the pmf value is above ε
        self.α = 0
        state = 1
        self.vals = {}
        summer = 0

        while(1):
            if state == 1:
                temp = poisson.pmf(self.α, self.λ)
                if(temp <= ε):
                    self.α += 1
                else:
                    self.vals[self.α] = temp
                    summer += temp
                    self.β = self.α+1
                    state = 2
            elif state == 2:
                temp = poisson.pmf(self.β, self.λ)
                if(temp > ε):
                    self.vals[self.β] = temp
                    summer += temp
                    self.β += 1
                else:
                    break

        # normalizing the pmf, values of n outside of [α, β] have pmf = 0

        added_val = (1-summer)/(self.β-self.α)
        for key in self.vals:
            self.vals[key] += added_val

    def f(self, n):
        try:
            Ret_value = self.vals[n]
        except(KeyError):
            Ret_value = 0
        finally:
            return Ret_value


class Beds():
    def __init__(self, requestVal, returnVal) -> None:
        self.requestVal = requestVal
        self.returnVal = returnVal
        self.poissonRequestVal = poisson_(self.requestVal)
        self.poissonReturnVal = poisson_(self.returnVal)


class Agent():

    def __init__(self, hospital, value, policy, eValue) -> None:
        self.hospital = hospital
        self.value = value
        self.policy = policy
        self.eValue = eValue

    def cost(self, A, B, C, D, bedsMoved):
        costVal = 0
        if A > B:
            costVal += -1*(A - B)*10
        if C > D:
            costVal += -1*(C - D)*20
        if bedsMoved > 0:
            costVal += bedsMoved*5
        return costVal

    def expectedReward(self, state, action):
        self.hospital.initialBedsNormal = self.hospital.availableBeds[0]
        self.hospital.initialBedsCovid = self.hospital.availableBeds[1]
        initialReward = 0
        new_state = [min(max((state[0] - action), 0), self.hospital.N),
                     min(max((state[1] + action), 0), self.hospital.N)]
        if action > 0:
            initialReward += -1*self.cost(0, 0, 0, 0, state[0] - action)
        self.hospital.availableBeds[0] = new_state[0]
        self.hospital.availableBeds[1] = new_state[1]

        for _ in range(7):
            A = Beds(
                self.hospital.availableBeds[0], max(min(self.hospital.initialBedsNormal - self.hospital.availableBeds[0], 0), self.hospital.N))
            B = Beds(
                self.hospital.availableBeds[1], max(min(self.hospital.initialBedsCovid - self.hospital.availableBeds[1], 0), self.hospital.N))

            for Aini in range(A.poissonRequestVal.α, A.poissonRequestVal.β):
                for Bini in range(B.poissonRequestVal.α, B.poissonRequestVal.β):
                    for Afin in range(A.poissonReturnVal.α, A.poissonReturnVal.β):
                        for Bfin in range(B.poissonReturnVal.α, B.poissonReturnVal.β):
                            probability = A.poissonRequestVal.vals[Aini] * B.poissonRequestVal.vals[Bini] * \
                                A.poissonReturnVal.vals[Afin] * \
                                B.poissonReturnVal.vals[Bfin]
                            rnew = 0
                            if Aini > new_state[0]:
                                rnew += self.cost(Aini, new_state[0], 0, 0, 0)
                            if Bini > new_state[1]:
                                rnew += self.cost(Bini, new_state[1], 0, 0, 0)

                            returnA = Afin
                            returnB = Bfin
                            new_state = [0, 0]
                            new_state[0] = min(max(
                                new_state[0] - Aini + returnA, 0), self.hospital.N)
                            new_state[1] = min(max(
                                new_state[1] - Bini + returnB, 0), self.hospital.N)

                            initialReward += probability * \
                                (rnew + self.hospital.discountRate *
                                 self.value[new_state[0]][new_state[1]])
            self.hospital.availableBeds[0] = new_state[0]
            self.hospital.availableBeds[1] = new_state[1]
        return initialReward

    def policy_improvement(self):
        policy_stable = True
        for i in range(self.value.shape[0]):
            for j in range(self.value.shape[1]):
                old_action = self.policy[i][j]

                max_act_val = None
                max_act = None

                minPossibleAction = self.hospital.availableBeds[1]
                maxPossibleAction = self.hospital.availableBeds[0]

                for act in range(-1*minPossibleAction, maxPossibleAction):
                    sigma = self.expectedReward([i, j], act)
                    if max_act_val == None:
                        max_act_val = sigma
                        max_act = act
                    elif max_act_val < sigma:
                        max_act = act

                self.policy[i][j] = max_act
                if old_action != self.policy[i][j]:
                    policy_stable = False
        return policy_stable


hospital = Hospital(np.array([5, 5]), 5, 5, 10, 0.8)
agent = Agent(hospital, np.zeros((hospital.N+1, hospital.N+1)),
              np.zeros((hospital.N+1, hospital.N+1)), 50)

while(1):
    p = agent.policy_improvement()
    if p == True:
        break
