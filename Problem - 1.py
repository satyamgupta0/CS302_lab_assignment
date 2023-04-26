# Importing numpy as np
import numpy as np
import random
# Setting printoptions so that there will be no e in the result
np.set_printoptions(suppress=True, precision=15)

# Creating a class ValueIteration()

# Method - 1 : Assuming the agent takes non deterministic action at each state


class ValueIteration():
    # With the constructor initialzing values
    def __init__(self, states, values, policy, rewards, actions, probabilites, futureDecay, theta, actionArray) -> None:
        self.states = states
        self.values = values
        self.policy = policy
        self.rewards = rewards
        self.actions = actions
        self.probabilites = probabilites
        self.futureDecay = futureDecay
        self.theta = theta
        self.actionArray = actionArray

    # Defining run function which keeps the agent running until target is achieved
    def run(self):
        while True:
            # Assuming that agent takes two actions either to "CONTINUE" or to "QUIT"
            priorValues = self.values.copy()
            for state in self.states:
                Q_Val = [0]*2
                # Calculating the reward obtained till now
                rewardObtainedTill = 0
                for i in range(state):
                    rewardObtainedTill += self.rewards[i]
                for action in self.actions:
                    # If the user opts to "CONTINUE" then as the environment is stochastic and the agent can end in  correct answer or wrong answer
                    # So, with probability of correctly answering the question he will get reward of the current state and
                    # Using discount rate for the future alignment
                    # If the agent end up in answering the question wrong then with that probability removing the reward which he got till now
                    if action == "CONTINUE":
                        Q_Val[0] = self.probabilites[state]*(self.rewards[state] + self.futureDecay*self.values[state+1]) + (
                            1-self.probabilites[state])*(-1*rewardObtainedTill)
                    # If agent opts to quit then giving him the entire reward which he got till now
                    else:
                        Q_Val[1] = rewardObtainedTill

                self.values[state] = max(Q_Val)
            if(np.allclose(self.values, priorValues, atol=self.theta)):
                break

    # Finding policy for the best action possible
    def getPolicy(self):
        for state in self.states:
            Q_A = [0]*2
            rewardObtainedTill = 0
            for i in range(state):
                rewardObtainedTill += self.rewards[i]
            for action in self.actions:
                if action == "CONTINUE":
                    Q_A[0] = self.probabilites[state]*(self.rewards[state] + self.futureDecay*self.values[state+1]) + (
                        1-self.probabilites[state])*(-1*rewardObtainedTill)
                else:
                    Q_A[1] = rewardObtainedTill

            maxVal = np.argmax(Q_A)
            if maxVal == 0:
                self.actionArray[state] = "CONTINUE"
            else:
                self.actionArray[state] = "QUIT"


# Method - 2: Assuming that the agent takes only deterministic action and it is to continue
class ValueIteration1():
    def __init__(self, states, values, policy, rewards, actions, probabilites, futureDecay, theta) -> None:
        self.states = states
        self.values = values
        self.policy = policy
        self.rewards = rewards
        self.actions = actions
        self.probabilites = probabilites
        self.futureDecay = futureDecay
        self.theta = theta

    def run(self):
        while True:
            priorValues = self.values.copy()
            for state in self.states:
                rewardObtainedTill = 0
                for i in range(state):
                    rewardObtainedTill += self.rewards[i]
                for action in self.actions:
                    if action == "CONTINUE":
                        self.values[state] = self.probabilites[state]*(self.rewards[state] + self.futureDecay*(
                            self.values[state+1])) + (1-self.probabilites[state])*(-1*(rewardObtainedTill))
            if(np.allclose(self.values, priorValues, atol=self.theta)):
                break


class SARS():
    def __init__(self, rewards, probabilites) -> None:
        self.rewards = rewards
        self.probabilites = probabilites
        self.total = 0

    def run(self):
        totalReward = 0
        for _ in range(10000):
            currReward = 0
            for ques in range(len(self.rewards)):
                randomNumber = random.random()
                if randomNumber > self.probabilites[ques]:
                    break
                else:
                    currReward += self.rewards[ques]
            totalReward += currReward
        self.total = totalReward / 10000


agent1 = ValueIteration1(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), np.zeros(11), np.zeros(10), np.array([100, 500, 1000, 5000, 10000, 50000,
                                                                                                         100000, 500000, 1000000, 5000000]), ["CONTINUE"], np.array([0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]), 0.7, 1e-12)

agent2 = SARS(np.array([100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000]), np.array(
    [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]))
agent = ValueIteration(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), np.zeros(11), np.zeros(10), np.array([100, 500, 1000, 5000, 10000, 50000,
                                                                                                       100000, 500000, 1000000, 5000000]), ["CONTINUE", "QUIT"], np.array([0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]), 0.5, 1e-12, ["", "", "", "", "", "", "", "", "", ""])

print("========================================================================================")
agent.run()
agent.getPolicy()
print("For max reward the agent should follow the policy : ", agent.actionArray)
print("========================================================================================")
agent1.run()
print("For max reward the agent should quit at " +
      str(np.argmax(agent1.values) + 1) + " state.")
print("========================================================================================")
agent2.run()

print("The optimal value using SARS is " + str(agent2.total) +
      " and using value iteration it is " + str(agent.values[np.argmax(agent1.values)]))
print("========================================================================================")
