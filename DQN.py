import gym
import numpy as np

INPUT_SIZE = 2

class ReplayBuffer:
    """Um buffer que armazena transições e permite a amostragem de transições aleatórias."""

    def __init__(self, max_size):
        """Cria um replay buffer.

        Args:
            max_size (int): número máximo de transições armazenadas pelo buffer.
        """
        self.max_size = max_size
        self.transition_counter = 0

        self.states = np.zeros((self.max_size, INPUT_SIZE), dtype = np.float32)
        self.states2 = np.zeros((self.max_size, INPUT_SIZE), dtype = np.float32)

        self.rewards = np.zeros(self.max_size, dtype = np.float32)
        self.actions = np.zeros(self.max_size, dtype = np.int32)
        self.terminal = np.zeros(self.max_size, dtype = np.int32)

    def add_transition(self, transition):
        """Adiciona uma transição ao replay buffer.

        Args:
            transition (tuple): uma tupla representando a transição no formato (estado atual, estado seguinte, recompensa, acao, terminado).
            estado atual e estado seguinte são re
        """
        index = self.transition_counter % self.max_size

        self.states[index] = transition[0]
        self.states2[index] = transition[1]
        self.rewards[index] = transition[2]
        self.actions[index] = transition[3]
        self.terminal[index] = transition[4]

        self.transition_counter += 1

    def sample_transitions(self, num_samples):
        """Retorna uma lista com transições amostradas aleatoriamente.

        Args:
            num_samples (int): o número de transições desejadas.
        """
        index_range = min(self.transition_counter, self.max_size)
        sample = np.random.choice(index_range)

        states = self.states[sample]
        states2 = self.states[sample]
        rewards = self.rewards[sample]
        actions = self.actions[sample]
        terminal = self.termina[sample]

        return states, states2, rewards, actions, terminal

    def get_size(self):
        """Retorna o número de transições armazenadas pelo buffer."""
        return min(self.transition_counter, self.max_size)

    def get_max_size(self):
        """Retorna o número máximo de transições armazenadas pelo buffer."""
        return self.max_size


class DQNAgent:
    """Implementa um agente de RL usando Deep Q-Learning."""

    def __init__(self, state_dim, action_dim, architecture,
                 buffer_size=100_000,
                 batch_size=128,
                 gamma=1.00):
        """Cria um agente de DQN com os hiperparâmetros especificados

        Args:
            state_dim (int): número de variáveis de estado.
            action_dim (int): número de ações possíveis.
            architecture (list of float, optional): lista com o número de neurônios
                                                    de cada camada da DQN.
            buffer_size (int, optional): tamanho máximo do replay buffer.
            batch_size (int, optional): número de transições utilizadas por batch.
            gamma (float, optional): fator de desconto utilizado no calculo do retorno.
        """
        pass

    def act(self, state, epsilon=0):
        """Retorna a ação tomada pelo agente no estado `state`.

        Args:
            state: o estado do ambiente.
            epsilon (int, optional): o valor de epsilon a ser considerado
                                     na política epsilon-greedy.
        """
        pass

    def optimize(self):
        """Roda um passo de otimização da DQN.

        Obtém `self.batch_size` transições do replay buffer
        e treina a rede neural. Se o replay buffer não tiver
        transições suficientes, a função não deve fazer nada.
        """
        pass


if __name__ == '__main__':
    # Crie o ambiente 'pong:turing-easy-v0'
    env = ...

    # Hiperparâmetros da política epsilon-greedy
    initial_eps = 1
    min_eps = 0.01
    eps_decay = .85
    eps = initial_eps
    gamma = .995

    # Número total de episódios
    num_episodes = 25

    agent = DQNAgent(action_dim=...,
                     state_dim=...,
                     architecture=[32, 32],
                     batch_size=512,
                     gamma=gamma)

    for episode in range(num_episodes):
        # Rodar o episódio, da mesma forma que com o agente aleatório.
        pass

    # Fechando o ambiente
    env.close()
