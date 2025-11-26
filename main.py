import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

###########################################################################################################################################################
######################################################################## QUESTÃO 1 ########################################################################
###########################################################################################################################################################

N = 10000

k_ = 20

p = k_/N


#network = nx.gnp_random_graph(N, p)


def run_sis_simulation(G, beta, mu, t_max=100):
    N = G.number_of_nodes()
    
    status = np.zeros(N, dtype=int)
    
    initial_infected = np.random.choice(range(N), 5, replace=False)
    status[initial_infected] = 1
    
    history = []
    
    for t in range(t_max):
        num_infected = np.sum(status)
        history.append(num_infected)
        
        if num_infected == 0:
            history.extend([0] * (t_max - t - 1))
            break
            
        next_status = status.copy()
        
        infected_nodes = np.where(status == 1)[0]
        
        for i in infected_nodes:
            if np.random.random() < mu:
                next_status[i] = 0 # voltando a ser suscetível
            
            for neighbor in G.neighbors(i):
                if status[neighbor] == 0:
                    if np.random.random() < beta:
                        next_status[neighbor] = 1

        status = next_status

    return history

# # situação a
# beta = 0.02
# mu = 0.1

# resultados_a = []

# for each in range(100):
#     hist = run_sis_simulation(network, beta, mu)
#     resultados_a.append(hist)

# media_a = np.mean(resultados_a, axis=0)

# #situação b
# mu = 0.4

# resultados_b = []

# for each in range(100):
#     hist = run_sis_simulation(network, beta, mu)
#     resultados_b.append(hist)
    

# media_b = np.mean(resultados_b, axis=0)

# #situação c
# mu = 0.5


# resultados_c = []

# for each in range(100):
#     hist = run_sis_simulation(network, beta, mu)
#     resultados_c.append(hist)
    

# media_c = np.mean(resultados_c, axis=0)


#plot exercicio 1
# plt.figure(figsize=(10, 6))
# plt.title("Evolução da Epidemia em Rede Aleatória (ER)")
# plt.xlabel("Tempo (t)")
# plt.ylabel("Número Médio de Infectados")

# tempo = range(len(media_a))

# plt.plot(tempo, media_a, label=f'Cenário A (β = 0.02, μ = 0.1)', linewidth=2)
# plt.plot(tempo, media_b, label=f'Cenário B (β = 0.02, μ = 0.4)', linewidth=2, color='orange')
# plt.plot(tempo, media_c, label=f'Cenário C (β = 0.02, μ = 0.5)', linewidth=2, color='yellow')

# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

###########################################################################################################################################################
######################################################################## QUESTÃO 2 ########################################################################
###########################################################################################################################################################

N = 10000
gamma = 2.5
k_ = 20

def generate_scale_free_network(N, gamma, target_k):
    """
    Gera uma rede livre de escala usando o Modelo de Configuração.
    
    Parâmetros:
    N: Número de nós 
    gamma: Expoente da lei de potência 
    target_k: Grau médio desejado 
    """
    
    k_min = target_k * (gamma - 2) / (gamma - 1)
    
    u = np.random.uniform(0, 1, N)
    
    degrees = k_min * (1 - u)**(1 / (1 - gamma))
    
    degrees = np.round(degrees).astype(int)
    degrees = np.maximum(degrees, int(k_min)) 
    
    if np.sum(degrees) % 2 != 0:    
        idx = np.random.randint(0, N)
        degrees[idx] += 1
    G_multi = nx.configuration_model(degrees)
    
    G_simple = nx.Graph(G_multi)
    G_simple.remove_edges_from(nx.selfloop_edges(G_simple))
    
    return G_simple

# network_sf = generate_scale_free_network(N, gamma, k_)

# beta_q2 = 0.01

# #situação a
# mu = 0.1
# hist_a = []

# for _ in range(10): 
#     hist_a.append(run_sis_simulation(network_sf, beta_q2, mu))

# media_sf_a = np.mean(hist_a, axis=0)

# #situação b
# mu=0.2
# hist_b = []

# for _ in range(10): 
#     hist_b.append(run_sis_simulation(network_sf, beta_q2, mu))

# media_sf_b = np.mean(hist_b, axis=0)


# #situação c
# mu = 0.3
# hist_c = []

# for _ in range(10): 
#     hist_c.append(run_sis_simulation(network_sf, beta_q2, mu))

# media_sf_c = np.mean(hist_c, axis=0)

# tempo = range(len(media_sf_a))


# #plot exercicio 2

# plt.figure(figsize=(10, 6))
# plt.title("Evolução da Epidemia em Rede Livre de Escala (Scale Free)")
# plt.xlabel("Tempo (t)")
# plt.ylabel("Número Médio de Infectados")

# plt.plot(tempo, media_sf_a, label=f'Cenário A (β = 0.01, μ = 0.1)', linewidth=2)
# plt.plot(tempo, media_sf_b, label=f'Cenário B (β = 0.01, μ = 0.2)', linewidth=2, color='orange')
# plt.plot(tempo, media_sf_c, label=f'Cenário C (β = 0.01, μ = 0.3)', linewidth=2, color='yellow')

# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

###########################################################################################################################################################
######################################################################## QUESTÃO 3 ########################################################################
###########################################################################################################################################################

# funcoes auxliares
def select_random_immunization(G, fraction):
    N = G.number_of_nodes()
    num_immune = int(N * fraction)
    return np.random.choice(range(N), num_immune, replace=False)

def select_hub_immunization(G, fraction):
    N = G.number_of_nodes()
    num_immune = int(N * fraction)
    sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)
    top_hubs = [node for node, degree in sorted_nodes[:num_immune]]
    return np.array(top_hubs)

def select_neighbor_immunization(G, fraction):
    N = G.number_of_nodes()
    num_immune = int(N * fraction)
    immune_nodes = set()
    
    while len(immune_nodes) < num_immune:
        random_node = np.random.randint(0, N)
        neighbors = list(G.neighbors(random_node))
        
        if len(neighbors) > 0:
            neighbor_to_vaccinate = np.random.choice(neighbors)
            immune_nodes.add(neighbor_to_vaccinate)
            
    return np.array(list(immune_nodes))

def run_sis_simulation_immune(G, beta, mu, immunized_nodes, t_max=100):
    N = G.number_of_nodes()
    status = np.zeros(N, dtype=int)
    
    status[immunized_nodes] = 2
    
    susceptible_nodes = np.where(status == 0)[0]
    if len(susceptible_nodes) >= 5:
        initial_infected = np.random.choice(susceptible_nodes, 5, replace=False)
        status[initial_infected] = 1
    
    history = []
    
    for t in range(t_max):
        num_infected = np.sum(status == 1)
        history.append(num_infected)
        
        if num_infected == 0:
            history.extend([0] * (t_max - t - 1))
            break
            
        next_status = status.copy()
        infected_nodes = np.where(status == 1)[0]
        
        for i in infected_nodes:
            if np.random.random() < mu:
                next_status[i] = 0 
            
            for neighbor in G.neighbors(i):
                if status[neighbor] == 0:
                    if np.random.random() < beta:
                        next_status[neighbor] = 1

        status = next_status

    return history



network_sf = generate_scale_free_network(N, gamma, k_)

fracoes = [each/10 for each in range(1,10)] 
beta_q3 = 0.01
mu_q3 = 0.1


resultado_random = []
resultado_hub = []
resultado_neighbor = []

for f in fracoes:    
    # aleatória
    imunes = select_random_immunization(network_sf, f)
    hist = run_sis_simulation_immune(network_sf, beta_q3, mu_q3, imunes)
    resultado_random.append(hist[-1]) 
    
    # hubs
    imunes = select_hub_immunization(network_sf, f)
    hist = run_sis_simulation_immune(network_sf, beta_q3, mu_q3, imunes)
    resultado_hub.append(hist[-1])
    
    # vizinhos
    imunes = select_neighbor_immunization(network_sf, f)
    hist = run_sis_simulation_immune(network_sf, beta_q3, mu_q3, imunes)
    resultado_neighbor.append(hist[-1])

# plot questao 3
plt.figure(figsize=(10, 6))
plt.plot(fracoes, resultado_random, label='Aleatória')
plt.plot(fracoes, resultado_neighbor, label='Vizinhos')
plt.plot(fracoes, resultado_hub, label='Hubs')

plt.title("Eficácia das Estratégias de Imunização (Rede Scale-Free)")
plt.xlabel("Fração de Vacinados")
plt.ylabel("Número Final de Infectados")
plt.legend()
plt.grid(True)
plt.show()