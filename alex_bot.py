import torch
import random
import numpy as np
from collections import deque
from model import QTrainer, Linear_QNet



# java -jar JSettlersServer-2.0.00.jar -Djsettlers.bots.percent3p=25 -Djsettlers.bots.botgames.total=1000


MAX_MEMORY = 1000
LR = 0.001
THRESHOLD = 0.8 # THRESHOLD FOR ACCEPTING TRADE
BATCH_SIZE = 1000
NUM_GAMES = 10


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0
        self.action_list = []
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(22, 256, 1)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def save_model(self, model_name='model.pth'):
        self.model.save(model_name)

    def load_model(self, model_name='model.pth'):
        self.model.load(model_name)

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states)

    def train_short_memory(self, state, action, reward, next_state):
        self.trainer.train_step(state, action, reward, next_state)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        if random.randint(0, 200) < self.epsilon:
            print("Random:")
            action = random.randint(0, 1)
        else:
            print("Using Model:")
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            if prediction[0] > THRESHOLD:
                action = 1
            else:
                action = 0
        return action


def train():
    from server import JSettlersServer

    # TODO: plot these as needed.
    placements = [0,0,0,0]
    placement_history = deque(maxlen=10) # tracks last 5 games
    total_score = 0
    avg_placement = []



    agent = Agent()
    game = JSettlersServer("localhost", 2004, agent, timeout=120)
    while True:
        if agent.n_games == NUM_GAMES:
            break
        feat_vector = game.get_message()
        if feat_vector is None:
            # print("Msg skipped: ")
            if game.final_place != -1:
                agent.n_games += 1
                reward = 0
                if game.final_place == 1:
                    reward = 10
                    placements[0] += 1 
                    placement_history.append(1)
                elif game.final_place == 2:
                    reward = 5
                    placements[1] += 1
                    placement_history.append(2)
                elif game.final_place == 3:
                    reward = -3
                    placements[2] +=1 
                    placement_history.append(3)
                elif game.final_place == 4:
                    reward = -6
                    placements[3] +=1 
                    placement_history.append(4)
                else:
                    print("game not finished yet")

                for elem in agent.action_list:
                    agent.remember(elem[0], elem[1], elem[2] + reward, elem[3])

                print("Placed " + str(game.final_place))

                agent.action_list.clear()
                

                agent.train_long_memory()
                print("Finished training long term memory")
                agent.model.save()

                agent.n_games += 1

                placement_history.append(game.final_place)

                avg_placement.append(sum(placement_history) / len(placement_history))
                with open('placements.txt', 'a') as file:
                    file.write(str(game.final_place) + "\n")
                    # print("    " + str(game.final_place))

                game.reset() 


        else:
            cur_state = feat_vector

            action = agent.get_action(feat_vector)

            print("Action: " + str(action))
            reward = game.play_step('trade', action)

            my_res = feat_vector[2:7]
            opp_res = feat_vector[7:12]
            get = feat_vector[12:17]
            give = feat_vector[17:22]
            if action == 0:
                new_state = np.array(feat_vector[0:12].tolist() + [0 for i in range(10)])
            else:
                my_new_res = my_res - give + get
                opp_new_res = opp_res - get + give
                new_state = np.array(feat_vector[0:2].tolist() + my_new_res.tolist() + opp_new_res.tolist() + [0 for i in range(10)])

            reward += (sum(get) - sum(give)) * 0.2
            agent.action_list.append((cur_state, action, reward, new_state))
            agent.train_short_memory(cur_state, action, reward, new_state)


if __name__ == '__main__':
    train()