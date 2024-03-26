import socket
import os
import numpy as np
from collections import deque
import random
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn



OBSERVATION_SPACE_SIZE = (1,22)
ACTION_SPACE_SIZE = 2
DISCOUNT = 0.99

REPLAY_MEMORY_SIZE = 100  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 16  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 8  # How many steps (samples) to use for training

UPDATE_TARGET_EVERY = 4  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = 0  # For model save

#  Stats settings
AGGREGATE_STATS_EVERY = 2  # episodes


class TradingNN(nn.Module):
    def __init__(self):
        super(TradingNN, self).__init__()
        self.input_size = 22
        self.hidden_size = 22
        self.output_size = 1

        # Define layers
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class DQNAgent:
    def __init__(self):
        # Main Model - used to actually fit
        self.model = self.create_model()
        self.history = None

        # Target model - used to predict, updated every so episodes or epochs
        # self.target_model = self.create_model()
        # self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE) # Used to create 'batches' for fitting

        # TODO: fix tensorboard
        # self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0 # Tracking how many more examples to see before updating target_model

        # Exploration settings
        self.epsilon = 1  # not a constant, going to be decayed
        self.EPSILON_DECAY = 0.975
        self.MIN_EPSILON = 0.001



    def create_model(self):
        self.model = TradingNN()

        # TODO: Debuggin line here
        print(self.model.summary())




    def update_replay_memory(self, transition):
        # Update replay_memory with new (state, action, reward, new_state, done)
        self.replay_memory.append(transition)


    def get_qs(self, terminal_state):
        return 1
        # return self.model.predict(np.array(terminal_state).reshape(-1, *terminal_state.shape))[0]

    def train(self, terminal_state):
        if len(self.replay_memory) < MINIBATCH_SIZE:
            return

        batch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        



class JSettlersServer:

    def __init__(self, host, port, dqnagent, timeout=None):
        # Used for training agent
        self.agent = dqnagent
        self.host = host
        self.port = port
        self.timeout = timeout
        self.prev_vector = None
        self.last_action = None

        #Used for logging models and stats
        self.ep_rewards = [0]
        self.curr_episode = 1
        self.standing_log = "agent_standings.csv"
        self.standing_results = [0,0,0,0]


    def run(self):
        soc = socket.socket()         # Create a socket object
        if self.timeout:
            soc.settimeout(self.timeout)
        try:
            print(str(self.host) + " " + str(self.port))
            soc.bind((self.host, self.port))

        except socket.error as err:
            print('Bind failed. Error Code : ' .format(err))

        soc.listen(10)
        print("Socket Listening ... ")
        while True:
            try:
                conn, addr = soc.accept()     # Establish connection with client.
                length_of_message = int.from_bytes(conn.recv(2), byteorder='big')
                msg = conn.recv(length_of_message).decode("UTF-8")
                print("Considering Trade ... ")
                action = self.handle_msg(msg)
                conn.send((str(action) + '\n').encode(encoding='UTF-8'))
                print("Result: " + str(action) + "\n")
            except socket.timeout:
                print("Timeout or error occured. Exiting ... ")
                break

    def get_action(self, state):
        state = np.array(state)
        state = state.reshape((1, 22))
        if np.random.random() > self.agent.epsilon:
            action = np.argmax(self.agent.get_qs(state))
        else:
            action = np.random.randint(0, ACTION_SPACE_SIZE)
        return action



    def handle_msg(self, msg):
        # self.agent.tensorboard.step = self.curr_episode
        print("Episode: ", self.curr_episode)
        msg_args = msg.split("|")


        print(msg_args[0])


        if msg_args[0] == "trade": #We're still playing a game; update our agent based on the rewards returned and take an action
            my_vp = int(msg_args[1])
            opp_vp = int(msg_args[2])
            my_res = [int(x) for x in msg_args[3].split(",")]
            opp_res = [int(x) for x in msg_args[4].split(",")]
            get = [int(x) for x in msg_args[5].split(",")]
            give = [int(x) for x in msg_args[6].split(",")]
            #Construct total feature vector
            feat_vector = np.array([my_vp] + [opp_vp] + my_res + opp_res + get + give)

            if self.prev_vector is not None:    # If we have a previous state, run a train step on the agent for the last action taken
                self.agent.update_replay_memory((self.prev_vector, self.last_action, 0, feat_vector, False))
                self.agent.train(False)
            else:
                print("First step. Ignoring training ... ")
            # Update actions so that on the next step, we'll train on these actions
            action = self.get_action(feat_vector)
            self.prev_vector = feat_vector
            self.last_action = action
            return action

        elif msg_args[0] == "end": #The game has ended, update our agent based on the rewards, update our logs, and reset for the next game
            is_over = str(msg_args[1])
            print("Result: ", is_over)
            if "true" in is_over:
                final_placing = int(msg_args[2])
                print("Game end. Final Placing: " + str(final_placing) + "\n\n")
                if (final_placing == 1):
                    reward = 10
                elif (final_placing == 2):
                    reward = 7
                if (final_placing == 3):
                    reward = 4
                elif (final_placing == 4):
                    reward = 0




            else:
                print("Unfinished game; ignoring result ...\n\n")

            return None


    def write_result(self, place):
        self.standing_results[place-1] += 1
        with open(self.standing_log, "w+") as f:
            for res in self.standing_results:
                f.write(str(res) + '\n')




if __name__ == "__main__":
    dqnagent = DQNAgent()
    server = JSettlersServer("localhost", 2004, dqnagent, timeout=120)
    server.run()