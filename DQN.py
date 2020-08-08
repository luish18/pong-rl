import gym
import numpy as np
from random import random
import torch 
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

INPUT_SIZE = 2

class ReplayBuffer:
    """Um buffer que armazena transições e permite a amostragem de transições aleatórias."""

    def __init__(self, max_size):
        """Cria um replay buffer.

        Args:
            max_size (int): número máximo de transições armazenadas pelo buffer.
        """
        self._max_size = max_size
        self._transition_counter = 0
        self._index = 0

        self._states = np.zeros((self._max_size, INPUT_SIZE), dtype = np.float32)
        self._states2 = np.zeros((self._max_size, INPUT_SIZE), dtype = np.float32)

        self._rewards = np.zeros(self._max_size, dtype = np.float32)
        self._actions = np.zeros(self._max_size, dtype = np.int32)
        self._terminal = np.zeros(self._max_size, dtype = np.int32)

    def add_transition(self, transition):
        """Adiciona uma transição ao replay buffer.

        Args:
            transition (tuple): uma tupla representando a transição no formato (estado atual, estado seguinte, recompensa, acao, terminado).
        """

        self._states[self._index] = transition[0]
        self._states2[self._index] = transition[1]
        self._rewards[self._index] = transition[2]
        self._actions[self._index] = transition[3]
        self._terminal[self._index] = transition[4]

        self._index = (self._index + 1)%self._max_size

        self._transition_counter += 1

    def sample_transitions(self, num_samples):
        """Retorna uma lista com transições amostradas aleatoriamente.

        Args:
            num_samples (int): o número de transições desejadas.
        """
        index_range = range(min(self._transition_counter, self._max_size))
        
        indexes = np.zeros(num_samples, dtype=np.int32)

        for i in range(num_samples):

            indexes[i] = np.random.choice(index_range)

       # print(f"\ntransitions_counter = {self._transition_counter}; index = {index}; transition = {transition}; len = {len(self._states)}")
        sample_list = [[self._states[index] for index in indexes], 
                       [self._states2[index] for index in indexes], 
                       [self._rewards[index] for index in indexes],
                       [self._actions[index] for index in indexes],
                       [self._terminal[index] for index in indexes]]
        

        return sample_list

    def get_size(self):
        """Retorna o número de transições armazenadas pelo buffer."""
        return min(self._transition_counter, self.max_size)

    def get_max_size(self):
        """Retorna o número máximo de transições armazenadas pelo buffer."""
        return self.max_size


class DQNAgent:
    """Implementa um agente de RL usando Deep Q-Learning."""

    def __init__(self, state_dim, action_dim, architecture,
                 buffer_size=100_000,
                 batch_size=128,
                 gamma=1.00,
                 alpha=0.005):
        """Cria um agente de DQN com os hiperparâmetros especificados

        Args:
            state_dim (int): número de variáveis de estado.
            action_dim (int): número de ações possíveis.
            architecture (list of float, optional): lista com o número de neurônios
                                                    de cada camada da DQN.(duas camadas)
            buffer_size (int, optional): tamanho máximo do replay buffer.
            batch_size (int, optional): número de transições utilizadas por batch.
            gamma (float, optional): fator de desconto utilizado no calculo do retorno.
            alpha (float, optional): learning rate
        """
        self.device = torch.device("cuda" if torch.cuda.is_available else "cpu")

        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._gamma = gamma  
        self._actions = action_dim
        self._states = state_dim
        self._alpha = alpha

        self._mem_replay = ReplayBuffer(self._buffer_size)

        #adiciona camada de entrada da rede
        self._dqn = nn.Sequential(nn.Linear(self._states, 32),
                                  nn.ReLU(),
                                  nn.Linear(32, 32),
                                  nn.ReLU(),
                                  nn.Linear(32,32),
                                  nn.ReLU(),
                                  nn.Linear(32, self._actions))

        self._optimizer = optim.Adam(self._dqn.parameters(), lr=self._alpha)

    def act(self, state, epsilon=0):
        """Retorna a ação tomada pelo agente no estado `state`.

        Args:
            state: o estado do ambiente.
            epsilon (int, optional): o valor de epsilon a ser considerado
                                     na política epsilon-greedy.
        """
        state = torch.Tensor(state)
        chance = random()

        if chance < epsilon:
            action = np.random.choice(range(self._actions))
            return [action, 0]
        else:
            value, index = self._dqn(state).squeeze(0).max(0)
            return [index.item(), value.item()]


    def optimize(self):
        """Roda um passo de otimização da DQN.

        Obtém `self.batch_size` transições do replay buffer
        e treina a rede neural. Se o replay buffer não tiver
        transições suficientes, a função não deve fazer nada.s
        """
        if self._buffer_size < self._batch_size:
            return

        batch = self._mem_replay.sample_transitions(self._batch_size)

        states = torch.Tensor(batch[0])
        states2 = torch.Tensor(batch[1])
        rewards = torch.tensor(batch[2])
        actions = torch.LongTensor(batch[3])
        terminal = torch.tensor(batch[4])

        
        q_previsto = torch.gather(self._dqn(states), 1, actions.unsqueeze(1)).squeeze(1)

        q_next = self._dqn(states2).max(1)[0].detach()

        target = rewards + self._gamma*q_next*(1-terminal)

        loss = F.mse_loss(q_previsto, target)

        self._optimizer.zero_grad()
        loss.backward()

        for param in self._dqn.parameters():
            param.grad.data.clamp_(-1,1)
        self._optimizer.step()

        return loss

    def add_to_replay(self, state, next_state, reward, action, terminal):

        self._mem_replay.add_transition((state, next_state, reward, action, terminal))



        
if __name__ == '__main__':
    # Crie o ambiente 'pong:turing-easy-v0'
    env = gym.make("pong:turing-easy-v0")

    # Hiperparâmetros da política epsilon-greedy
    initial_eps = 1
    min_eps = 0.001
    eps_decay = .95
    eps = initial_eps
    gamma = .995

    # Número total de episódios
    num_episodes = 100

    agent = DQNAgent(action_dim=3,
                     state_dim=2,
                     architecture=[32, 32],
                     batch_size=1024,
                     gamma=gamma,
                     alpha=0.005)
    wins = 0
    for episode in range(num_episodes):
        
        state = env.reset()
        ep_reward = 0
        q_value = 0
        steps = 0
        total_loss = 0
        done = False
        
        while not done:
            
            action, value = agent.act(state, eps)

            next_state, reward, done, info = env.step(action)
            agent.add_to_replay(state, next_state, reward, action, done)
            state = next_state

            agent.optimize()            

            steps += 1
            q_value += value
            ep_reward += reward

        if ep_reward > 0:
            wins += 1
        
        eps = max(eps*eps_decay, min_eps)
        print(f"episode {episode}|avg_q = {q_value/steps}|ep_reward = {ep_reward}|steps = {steps}|wins = {wins}")


    # Fechando o ambiente
    env.close()
