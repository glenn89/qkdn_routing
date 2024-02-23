import copy
import random

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import topology_conf


class QuantumEnvironment:
    def __init__(self):
        self.G = None
        self.topology_conf = None
        self.metric_type = 'qber'   # type: 'simple_shortest', 'weighted_shortest', 'qber', 'num_key', 'combination'
        self.num_seed = 0
        self.max_time_step = 0

        self.generate_key_size = 0
        self.generate_key_time_slot = 0
        self.init_qber = None
        self.key_pool_size = 0
        self.Key_pool = None
        self.key_life_time = 0
        self.time_step = 0

        self.session_blocking = 0
        self.total_generation_keys = 0
        self.remaining_keys = 0
        self.used_keys = 0

        self.reward = 0
        self.mean_value = 0
        self.std_deviation = 0
        self.count = 0
        self.cumulative_size = 1
        self.cumulative_edge_keys = None

    def generate_topology(self):
        self.G = nx.Graph()
        self.G.add_nodes_from(self.topology_conf['QKD_NODES'])
        # nx.set_node_attributes(self.G, self.topology_conf['QKD_NODES_NAME'], name="name")
        # nx.relabel_nodes(self.G, self.topology_conf['QKD_NODES_NAME'])

        edges = []
        node_1 = np.where(np.array(self.topology_conf['QKD_TOPOLOGY']) == 1)[0]
        node_2 = np.where(np.array(self.topology_conf['QKD_TOPOLOGY']) == 1)[1]

        for (i, j) in zip(node_1, node_2):
            edges.append((i, j))

        # Set edge's weight
        self.G.add_weighted_edges_from(list((edges[n][0], edges[n][1], 0) for n in range(len(edges))))
        edges_attribute = {
            (edges[n][0], edges[n][1]): {
                "count_rate": self.topology_conf['COUNT_RATE'][n],
                "init_qber": self.topology_conf['INIT_QBER'][n],
                "qber": self.topology_conf['QBER'][n],
                "num_key": self.topology_conf['num_key'][n]
            } for n in range(len(edges))
        }
        nx.set_edge_attributes(self.G, edges_attribute)
        self.key_pool.update((key, []) for key in self.G.edges)  # Generate key pool
        self.num_key_with_qber()  # Reflect the number of keys with qber

    def num_key_with_qber(self):
        edges = list(self.G.edges())

        # Update qber and count rate
        for edge in edges:
            # qber scope is 1% ~ 15%
            self.G[edge[0]][edge[1]]['qber'] = min(max(round(np.random.binomial(
                n=self.G[edge[0]][edge[1]]['count_rate'], p=self.G[edge[0]][edge[1]]['init_qber'] / 100
            ) / self.G[edge[0]][edge[1]]['count_rate'] * 100), 1), 15)

            if self.time_step != 0:
                self.G[edge[0]][edge[1]]['count_rate'] = np.random.randint(1, 5) * 100

        # Generate key with qber
        for edge in edges:
            error_rate = self.G[edge[0]][edge[1]]['qber'] / 100
            ############# Real key rate version #############
            try:
                generated_keys = round(
                    self.generate_key_size * max(
                        1 + error_rate * np.log2(error_rate) + (1 - error_rate) * np.log2(1 - error_rate), 0
                    )
                )
                if self.topology_conf['NAME'] == 'BUTTERFLY':
                    if edge[0] == 0 or edge[1] == 5:
                        generated_keys += 10
                self.total_generation_keys += generated_keys

                # Append key life time
                for _ in range(generated_keys):
                    if len(self.key_pool[edge]) + generated_keys > self.key_pool_size:
                        self.key_pool[edge] = self.key_pool[edge][len(self.key_pool[edge]) + generated_keys - self.key_pool_size:]
                    self.key_pool[edge].append(self.key_life_time)

                self.G[edge[0]][edge[1]]['num_key'] = len(self.key_pool[edge])

                
            except ValueError as e:
                print(error_rate, e)
            # generated_keys = self.generate_key_size
            # self.G[edge[0]][edge[1]]['num_key'] += generated_keys
            # self.total_generation_keys += generated_keys

        # if self.metric_type == 'simple_shortest':
        #     print(self.G.edges(data=True))

            ############# Each calcuation version #############
            # self.G[edge[0]][edge[1]]['num_key'] += sum(
            #     np.random.choice(
            #         [0, 1],
            #         size=self.generate_key_size,
            #         p=[error_rate, 1 - error_rate]
            #     )
            # )

        ############# Static version #############
        # self.G[edge[0]][edge[1]]['num_key'] += 5

    def plot_topology(self):
        position = {
            0: [2, 12], 1: [4, 14], 2: [4, 10], 3: [8, 14], 4: [8, 10], 5: [10, 12]
        }
        edge_labels = {}

        # nx.draw(self.G, pos=position, node_color=self.topology_conf['QKD_NODES_COLOR_MAP'], with_labels=True)
        nx.draw(self.G, with_labels=True)
        # labels = nx.get_edge_attributes(self.G, 'count_rate')
        for u, v, attr in self.G.edges(data=True):
            edge_labels[(u, v)] = "{0}/{1}\n{2}\n{3}".format(attr['qber'], attr['count_rate'], attr['num_key'], attr['weight'])
        # nx.draw_networkx_edge_labels(self.G, position, edge_labels=edge_labels)
        # nx.draw_networkx_edge_labels(self.G, edge_labels=edge_labels)
        plt.show()

    def plot_heatmap(self):
        # Plot the lower triangular part of the heatmap with a red colormap
        lower_triangular = np.tril(self.node_num_heat)
        plt.imshow(lower_triangular, cmap='hot_r', interpolation='nearest', alpha=0.7, vmin=0, vmax=1)
        plt.colorbar(label='Link Strength')

        # Set the background color to white for the parts that do not appear
        plt.gca().set_facecolor('white')

        plt.title('Sophisticated Network with Red Link Strength Heatmap')
        plt.show()

    def reset(self, seed, max_time_step):
        self.num_seed = seed
        np.random.seed(self.num_seed)
        self.max_time_step = max_time_step

        self.topology_conf = topology_conf.nsfnet_topo
        self.generate_key_time_slot = 20
        self.generate_key_size = 10
        self.consume_key_size = 4
        self.consume_mean = 4
        self.consume_std_dev = 2
        self.key_life_time = 30
        self.key_pool_size = 100_000
        self.key_pool = {}

        self.time_step = 0
        self.session_blocking = 0
        self.total_generation_keys = 0
        self.remaining_keys = 0
        self.used_keys = 0
        self.reward = 0
        self.alpha = 0.0001

        self.generate_topology()
        self.node_num_heat = np.zeros((len(self.G), len(self.G)))
        self.count = 0
        self.cumulative_size = 5
        self.cumulative_edge_keys = {}
        for edge in self.G.edges:
            self.cumulative_edge_keys[edge] = []

    def step(self):
        info = {}
        self.consume_key_size = max(int(np.random.normal(self.consume_mean, self.consume_std_dev)), 1)
        # self.consume_key_size = np.random.pareto(self.consume_mean, 1).astype(int)[0] + 3

        # Cumulative edge key at cumulative size for weighted average num key
        for edge in self.G.edges:
            if len(self.cumulative_edge_keys[edge]) < 6:
                self.cumulative_edge_keys[edge].append(self.G[edge[0]][edge[1]]['num_key'])
            else:
                self.cumulative_edge_keys[edge].pop(0)
                self.cumulative_edge_keys[edge].append(self.G[edge[0]][edge[1]]['num_key'])

        routing_path = self.find_routing_path()
        # if self.metric_type == 'num_key':
        #     print("time step: ", self.time_step, "routing path: ", routing_path)
            # self.plot_topology()

        if not routing_path:
            # print("Don't find the routing path")
            self.session_blocking -= 1
        else:
            # print("Find the routing path")
            self.reward += 1
            for i in range(len(routing_path) - 1):
                self.node_num_heat[routing_path[i]][routing_path[i+1]] += 1
                self.node_num_heat[routing_path[i+1]][routing_path[i]] += 1
                self.used_keys += self.consume_key_size
                if routing_path[i] < routing_path[i+1]:
                    self.key_pool[(routing_path[i], routing_path[i+1])] = self.key_pool[(routing_path[i], routing_path[i+1])][self.consume_key_size:]
                else:
                    self.key_pool[(routing_path[i+1], routing_path[i])] = self.key_pool[(routing_path[i+1], routing_path[i])][self.consume_key_size:]

        # for i in self.G.edges:
        #     if i == (0, 1):
        #         # if self.G.edges[i]['num_key'] != len(self.key_pool[i]):
        #         print("time step: ", self.time_step, "edge: ", i, "num_key: ", self.G.edges[i]['num_key'])
        #         print("time step: ", self.time_step, "edge: ", i, "num_key: ", len(self.key_pool[i]), self.key_pool[i])
        #         print()

        for u, v, attr in self.G.edges(data=True):
            self.remaining_keys += attr['num_key']
            self.key_pool[(u, v)] = [life - 1 for life in self.key_pool[(u, v)]]
            self.key_pool[(u, v)] = [life for life in self.key_pool[(u, v)] if life >= 1]
            self.G.edges[(u, v)]['num_key'] = len(self.key_pool[(u, v)])

        if self.time_step != 0 and self.time_step % self.generate_key_time_slot == 0:
            self.num_key_with_qber()
            # print(self.G.edges(data=True))
        # print(self.metric_type, self.G.edges(data=True))

        self.time_step += 1

        info = {
            'session_blocking': self.session_blocking,
            'total_generation_keys': self.total_generation_keys,
            'remaining_keys': self.remaining_keys,
            'used_keys': self.used_keys
        }

        return self.reward, info

    def temp_shrotest_path(self, current_node, target_node, weight):
        copied_G = copy.deepcopy(self.G)
        subnet = nx.subgraph_view(
            copied_G,
            filter_edge=lambda node_1_id, node_2_id: \
                True if copied_G.edges[(node_1_id, node_2_id)]['num_key'] >= self.consume_key_size else False
        )
        if len(subnet.edges) == 0 or not nx.has_path(subnet, source=current_node, target=target_node):
            return []

        for edge in subnet.edges:
            subnet[edge[0]][edge[1]]['weight'] = 1 / subnet[edge[0]][edge[1]]['num_key']
        routing_path = nx.astar_path(subnet, current_node, target_node, None, weight)
        # routing_path = nx.shortest_path(subnet, current_node, target_node)

        return routing_path

    def find_routing_path(self):
        # current_node, target_node = random.sample(range(0, self.topology_conf['NUM_QKD_NODE']), 2)
        current_node, target_node = np.random.choice(np.arange(0, self.topology_conf['NUM_QKD_NODE']), size=2, replace=False)
        # current_node, target_node = 0, 13
        accumulate_qber = []
        accumulate_num_key = []
        accumulate_count_rate = []
        loop_nodes = []
        routing_path = []
        routing_path.append(current_node)

        # Using weighted shortest path
        # routing_path = nx.shortest_path(self.G, source=0, target=5, weight='num_key')

        if self.metric_type == 'simple_shortest':
            routing_path = nx.shortest_path(self.G, current_node, target_node)
            for i in range(len(routing_path) - 1):
                if self.G[routing_path[i]][routing_path[i + 1]]['num_key'] < self.consume_key_size:
                    return []
            # copied_G = copy.deepcopy(self.G)
            # subnet = nx.subgraph_view(
            #     copied_G,
            #     filter_edge=lambda node_1_id, node_2_id: \
            #         True if copied_G.edges[(node_1_id, node_2_id)]['num_key'] >= self.consume_key_size else False
            # )
            # if len(subnet.edges) == 0 or not nx.has_path(subnet, source=current_node, target=target_node):
            #     return []
            #
            # # routing_path = nx.shortest_path(subnet, 0, 5)
            # for edge in subnet.edges:
            #     subnet[edge[0]][edge[1]]['weight'] = 1 / subnet[edge[0]][edge[1]]['num_key']
            # routing_path = nx.astar_path(subnet, current_node, target_node, None, 'weight')

        if self.metric_type == 'weighted_shortest':
            copied_G = copy.deepcopy(self.G)
            subnet = nx.subgraph_view(
                copied_G,
                filter_edge=lambda node_1_id, node_2_id: \
                    True if copied_G.edges[(node_1_id, node_2_id)]['num_key'] >= self.consume_key_size else False
            )
            if len(subnet.edges) == 0 or not nx.has_path(subnet, source=current_node, target=target_node):
                return []

            # routing_path = nx.shortest_path(subnet, 0, 5)
            for edge in subnet.edges:
                subnet[edge[0]][edge[1]]['weight'] = 1 / subnet[edge[0]][edge[1]]['num_key']
            routing_path = nx.shortest_path(subnet, current_node, target_node, 'weight')

        if self.metric_type == 'weighted_life_shortest':
            copied_G = copy.deepcopy(self.G)
            subnet = nx.subgraph_view(
                copied_G,
                filter_edge=lambda node_1_id, node_2_id: \
                    True if copied_G.edges[(node_1_id, node_2_id)]['num_key'] >= self.consume_key_size else False
            )
            if len(subnet.edges) == 0 or not nx.has_path(subnet, source=current_node, target=target_node):
                return []

            # routing_path = nx.shortest_path(subnet, 0, 5)
            for edge in subnet.edges:
                # life_time_weight = [self.key_life_time + 1 - key_life for key_life in self.key_pool[edge]]
                life_time_weight = [100 if key_life < self.key_life_time * 0.5 else 1 for key_life in self.key_pool[edge]]
                subnet[edge[0]][edge[1]]['weight'] = 1 / sum(life_time_weight)
            routing_path = nx.shortest_path(subnet, current_node, target_node, 'weight')

        # Using QBER
        if self.metric_type == 'qber':
            # shortest_routing_path = self.temp_shrotest_path(current_node, target_node, 'weight')
            while current_node != target_node:
                neighbor_nodes = [node for node in self.G.neighbors(current_node) if
                                  self.G[current_node][node]['num_key'] > 0 and
                                  self.G[current_node][node]['num_key'] >= self.consume_key_size]
                neighbor_nodes = [node for node in neighbor_nodes if node not in routing_path]  # check in routing path
                neighbor_nodes = [node for node in neighbor_nodes if node not in loop_nodes]  # check the loop

                if len(neighbor_nodes) == 0 and len(accumulate_count_rate) > 0:
                    loop_nodes.append(current_node)
                    routing_path.pop()
                    accumulate_qber.pop()
                    accumulate_count_rate.pop()
                    current_node = routing_path[-1]
                    continue

                elif len(neighbor_nodes) == 0 and len(accumulate_count_rate) == 0:
                    return []

                selected_node = self.select_next_node(current_node, neighbor_nodes, accumulate_qber, accumulate_num_key, accumulate_count_rate)
                routing_path.append(selected_node)
                current_node = selected_node

            # if len(shortest_routing_path) != 0 and len(shortest_routing_path) * 2 <= len(routing_path):
            #     routing_path = shortest_routing_path

        # Using Num_key
        if self.metric_type == 'num_key':
            # shortest_routing_path = self.temp_shrotest_path(current_node, target_node, 'weight')
            while current_node != target_node:
                neighbor_nodes = [node for node in self.G.neighbors(current_node) if
                                  self.G[current_node][node]['num_key'] > 0 and
                                  self.G[current_node][node]['num_key'] >= self.consume_key_size]
                neighbor_nodes = [node for node in neighbor_nodes if node not in routing_path]  # check in routing path
                neighbor_nodes = [node for node in neighbor_nodes if node not in loop_nodes]  # check the loop

                if len(neighbor_nodes) == 0 and len(accumulate_count_rate) > 0:
                    loop_nodes.append(current_node)
                    routing_path.pop()
                    accumulate_num_key.pop()
                    accumulate_count_rate.pop()
                    current_node = routing_path[-1]
                    continue

                elif len(neighbor_nodes) == 0 and len(accumulate_count_rate) == 0:
                    return []

                selected_node = self.select_next_node(current_node, neighbor_nodes, accumulate_qber, accumulate_num_key, accumulate_count_rate)
                routing_path.append(selected_node)
                current_node = selected_node

            # if len(shortest_routing_path) != 0 and len(shortest_routing_path) * 2 <= len(routing_path):
            #     routing_path = shortest_routing_path

        # Using QBER + Num_key
        if self.metric_type == 'combination':
            # shortest_routing_path = self.temp_shrotest_path(current_node, target_node, 'weight')
            while current_node != target_node:
                neighbor_nodes = [node for node in self.G.neighbors(current_node) if
                                  self.G[current_node][node]['num_key'] > 0 and
                                  self.G[current_node][node]['num_key'] >= self.consume_key_size]
                neighbor_nodes = [node for node in neighbor_nodes if node not in routing_path]  # check in routing path
                neighbor_nodes = [node for node in neighbor_nodes if node not in loop_nodes]  # check the loop

                if len(neighbor_nodes) == 0 and len(accumulate_count_rate) > 0:
                    loop_nodes.append(current_node)
                    routing_path.pop()
                    accumulate_qber.pop()
                    accumulate_num_key.pop()
                    accumulate_count_rate.pop()
                    current_node = routing_path[-1]
                    continue

                elif len(neighbor_nodes) == 0 and len(accumulate_count_rate) == 0:
                    return []

                selected_node = self.select_next_node(
                    current_node, neighbor_nodes,
                    accumulate_qber, accumulate_num_key, accumulate_count_rate
                )
                routing_path.append(selected_node)
                current_node = selected_node

            # print("!!!!!!!!!!", len(routing_path), routing_path, len(shortest_routing_path), shortest_routing_path)
            # if len(shortest_routing_path) != 0 and len(shortest_routing_path) * 2 <= len(routing_path):
            #     routing_path = shortest_routing_path
            #     self.count += 1
            # print(self.count)

        # if self.metric_type == 'combination':
        #     print("Final routing path: ", routing_path)
        # Consumed quantum key
        for i in range(len(routing_path) - 1):
            if routing_path[i] < routing_path[i+1]:
                self.G[routing_path[i]][routing_path[i+1]]['num_key'] -= self.consume_key_size
            else:
                self.G[routing_path[i+1]][routing_path[i]]['num_key'] -= self.consume_key_size

        return routing_path

    def select_next_node(self, current_node, neighbor_nodes, accumulate_qber, accumulate_num_key, accumulate_count_rate):
        current_edges = list(self.G.edges(current_node))
        for edge in current_edges:
            self.G[edge[0]][edge[1]]['weight'] = self.calculate_weight(
                edge, accumulate_qber, accumulate_num_key, accumulate_count_rate,
                self.G[edge[0]][edge[1]]['qber'],
                self.G[edge[0]][edge[1]]['num_key'],
                self.G[edge[0]][edge[1]]['count_rate']
            )

        min_weight_neighbor = min(neighbor_nodes,
                                  key=lambda neighbor: self.G[current_node][neighbor].get('weight', float('inf'))
                                  )
        accumulate_qber.append(self.G[current_node][min_weight_neighbor]['qber'])
        accumulate_num_key.append(self.G[current_node][min_weight_neighbor]['num_key'])
        accumulate_count_rate.append(self.G[current_node][min_weight_neighbor]['count_rate'])

        return min_weight_neighbor

    def calculate_weight(self, edge, accumulate_qber, accumulate_num_key, accumulate_count_rate, current_qber, current_num_key, current_count_rate):
        weight = 0
        qber_weight = 0
        num_key_weight = 0
        numerator_sum = 0
        denominator_sum = 0

        if self.metric_type == 'qber':
            if len(accumulate_count_rate) == 0:
                qber_weight = (current_qber * current_count_rate) / (current_count_rate)
            else:
                for i in range(len(accumulate_qber)):
                    numerator_sum += accumulate_qber[i] * accumulate_count_rate[i]
                numerator_sum += current_qber * current_count_rate
                denominator_sum = sum(accumulate_count_rate) + current_count_rate
                qber_weight = numerator_sum / denominator_sum
            weight = qber_weight

        if self.metric_type == 'num_key':
            if current_num_key == 0:
                current_num_key = 10000
            else:
                if edge[0] > edge[1]:
                    edge = (edge[1], edge[0])
                current_num_key = sum(self.cumulative_edge_keys[edge]) / len(self.cumulative_edge_keys[edge])
            if len(accumulate_count_rate) == 0:
                num_key_weight = 1 / current_num_key
            else:
                num_key_weight = 1 / current_num_key
                ######## Accumulate weight version ########
                # for i in range(len(accumulate_num_key)):
                #     denominator_sum += accumulate_num_key[i]
                # denominator_sum += current_num_key
                # numerator_sum = current_num_key
                # num_key_weight = numerator_sum / denominator_sum
            weight = num_key_weight

        if self.metric_type == 'combination':
            # Calculate QBER weight
            if len(accumulate_count_rate) == 0:
                qber_weight = (current_qber * current_count_rate) / current_count_rate
            else:
                for i in range(len(accumulate_qber)):
                    numerator_sum += accumulate_qber[i] * accumulate_count_rate[i]
                numerator_sum += current_qber * current_count_rate
                denominator_sum = sum(accumulate_count_rate) + current_count_rate
                qber_weight = numerator_sum / denominator_sum

            # Calculate num_key weight
            if current_num_key == 0:
                current_num_key = 10000
            else:
                if edge[0] > edge[1]:
                    edge = (edge[1], edge[0])
                current_num_key = sum(self.cumulative_edge_keys[edge]) / len(self.cumulative_edge_keys[edge])
            if len(accumulate_count_rate) == 0:
                num_key_weight = 1 / current_num_key
            else:
                num_key_weight = 1 / current_num_key
                # for i in range(len(accumulate_num_key)):
                #     denominator_sum += accumulate_num_key[i]
                # denominator_sum += current_num_key
                # numerator_sum = current_num_key
                # num_key_weight = numerator_sum / denominator_sum
            weight = self.alpha * qber_weight + (1 - self.alpha) * num_key_weight
            # print("!!!!!!!!!!!!!", self.alpha * qber_weight, (1 - self.alpha) * num_key_weight, weight)

        return weight


if __name__ == "__main__":
    env = QuantumEnvironment()
    max_time_step = 300
    num_simulation = 10
    seed = 0

    weighted_shortest_reward, shortest_reward, qber_reward, num_key_reward, combination_reward = 0, 0, 0, 0, 0
    weighted_shortest_average_reward, shortest_average_reward, qber_average_reward, num_key_average_reward, combination_average_reward = 0, 0, 0, 0, 0
    weighted_shortest_average_session_blocking, shortest_average_session_blocking, qber_average_session_blocking, num_key_average_session_blocking, combination_average_session_blocking = 0, 0, 0, 0, 0
    weighted_shortest_average_total_generation_keys, shortest_average_total_generation_keys, qber_average_total_generation_keys, num_key_average_total_generation_keys, combination_average_total_generation_keys = 0, 0, 0, 0, 0
    weighted_shortest_average_remaining_keys, shortest_average_remaining_keys, qber_average_remaining_keys, num_key_average_remaining_keys, combination_average_remaining_keys = 0, 0, 0, 0, 0
    weighted_shortest_average_used_keys, shortest_average_used_keys, qber_average_used_keys, num_key_average_used_keys, combination_average_used_keys = 0, 0, 0, 0, 0

    # Shortest path simulation
    env.metric_type = 'simple_shortest'
    env.reset(seed=seed, max_time_step=max_time_step)
    env.plot_topology()
    for _ in range(num_simulation):
        env.reset(seed=seed, max_time_step=max_time_step)
        for _ in range(max_time_step):
            shortest_reward, info = env.step()
        shortest_average_reward += shortest_reward
        shortest_average_session_blocking += info['session_blocking']
        shortest_average_total_generation_keys += info['total_generation_keys']
        shortest_average_remaining_keys += info['remaining_keys']
        shortest_average_used_keys += info['used_keys']
        seed += 1
    shortest_average_reward /= num_simulation
    shortest_average_session_blocking /= num_simulation
    shortest_average_total_generation_keys /= num_simulation
    shortest_average_remaining_keys /= num_simulation
    shortest_average_used_keys /= num_simulation
    seed = 0
    # env.plot_topology()
    # env.plot_heatmap()

    # Weighted shortest path simulation
    env.metric_type = 'weighted_shortest'
    env.reset(seed=seed, max_time_step=max_time_step)
    # env.plot_topology()
    for _ in range(num_simulation):
        env.reset(seed=seed, max_time_step=max_time_step)
        for _ in range(max_time_step):
            weighted_shortest_reward, info = env.step()
        weighted_shortest_average_reward += weighted_shortest_reward
        weighted_shortest_average_session_blocking += info['session_blocking']
        weighted_shortest_average_total_generation_keys += info['total_generation_keys']
        weighted_shortest_average_remaining_keys += info['remaining_keys']
        weighted_shortest_average_used_keys += info['used_keys']
        seed += 1
    weighted_shortest_average_reward /= num_simulation
    weighted_shortest_average_session_blocking /= num_simulation
    weighted_shortest_average_total_generation_keys /= num_simulation
    weighted_shortest_average_remaining_keys /= num_simulation
    weighted_shortest_average_used_keys /= num_simulation
    seed = 0
    # env.plot_topology()
    # env.plot_heatmap()

    # QBER simulation
    env.metric_type = 'weighted_life_shortest'
    env.reset(seed=seed, max_time_step=max_time_step)
    # env.plot_topology()
    for _ in range(num_simulation):
        env.reset(seed=seed, max_time_step=max_time_step)
        for _ in range(max_time_step):
            qber_reward, info = env.step()
        qber_average_reward += qber_reward
        qber_average_session_blocking += info['session_blocking']
        qber_average_total_generation_keys += info['total_generation_keys']
        qber_average_remaining_keys += info['remaining_keys']
        qber_average_used_keys += info['used_keys']
        seed += 1
    qber_average_reward /= num_simulation
    qber_average_session_blocking /= num_simulation
    qber_average_total_generation_keys /= num_simulation
    qber_average_remaining_keys /= num_simulation
    qber_average_used_keys /= num_simulation
    seed = 0
    # # env.plot_topology()
    # # env.plot_heatmap()
    #
    # # Num keys simulation
    # env.metric_type = 'num_key'
    # env.reset(seed=seed, max_time_step=max_time_step)
    # for _ in range(num_simulation):
    #     env.reset(seed=seed, max_time_step=max_time_step)
    #     for _ in range(max_time_step):
    #         num_key_reward, info = env.step()
    #     num_key_average_reward += num_key_reward
    #     num_key_average_session_blocking += info['session_blocking']
    #     num_key_average_total_generation_keys += info['total_generation_keys']
    #     num_key_average_remaining_keys += info['remaining_keys']
    #     num_key_average_used_keys += info['used_keys']
    #     seed += 1
    # num_key_average_reward /= num_simulation
    # num_key_average_session_blocking /= num_simulation
    # num_key_average_total_generation_keys /= num_simulation
    # num_key_average_remaining_keys /= num_simulation
    # num_key_average_used_keys /= num_simulation
    # seed = 0
    # # env.plot_topology()
    # # env.plot_heatmap()
    #
    # # QBER + Num keys simulation
    # env.metric_type = 'combination'
    # env.reset(seed=seed, max_time_step=max_time_step)
    # for _ in range(num_simulation):
    #     env.reset(seed=seed, max_time_step=max_time_step)
    #     for _ in range(max_time_step):
    #         combination_reward, info = env.step()
    #     combination_average_reward += combination_reward
    #     combination_average_session_blocking += info['session_blocking']
    #     combination_average_total_generation_keys += info['total_generation_keys']
    #     combination_average_remaining_keys += info['remaining_keys']
    #     combination_average_used_keys += info['used_keys']
    #     seed += 1
    # combination_average_reward /= num_simulation
    # combination_average_session_blocking /= num_simulation
    # combination_average_total_generation_keys /= num_simulation
    # combination_average_remaining_keys /= num_simulation
    # combination_average_used_keys /= num_simulation
    # seed = 0
    # env.plot_topology()
    # env.plot_heatmap()

    # print("QBER results: ", qber_reward, qber_session_blocking)
    # print("Num keys results: ", num_key_reward, num_key_session_blocking)
    # print("QBER + Num keys results: ", combination_reward, combination_session_blocking)

    # Print the results in a tabular format
    print("Simulation information")
    print("The number of max time step: ", max_time_step)
    print("The number of simulation: ", num_simulation)
    print()
    print("Average Results:")
    print(f"{'Metric':<20}{'Success':<10}{'Session Blocking':<20}{'Total generation keys':<25}{'Used keys':<20}{'Used percentage':<10}")
    print(f"{'simple_shortest':<20}{shortest_average_reward:<10}{shortest_average_session_blocking:<20}{shortest_average_total_generation_keys:<25}{shortest_average_used_keys:<20}{(shortest_average_used_keys/shortest_average_total_generation_keys) * 100:<4.2f}%")
    print(f"{'weighted_shortest':<20}{weighted_shortest_average_reward:<10}{weighted_shortest_average_session_blocking:<20}{weighted_shortest_average_total_generation_keys:<25}{weighted_shortest_average_used_keys:<20}{(weighted_shortest_average_used_keys / weighted_shortest_average_total_generation_keys) * 100:<4.2f}%")
    print(f"{'life_time_shortest':<20}{qber_average_reward:<10}{qber_average_session_blocking:<20}{qber_average_total_generation_keys:<25}{qber_average_used_keys:<20}{(qber_average_used_keys/qber_average_total_generation_keys) * 100:<4.2f}%")
    # print(f"{'Num keys':<20}{num_key_average_reward:<10}{num_key_average_session_blocking:<20}{num_key_average_total_generation_keys:<25}{num_key_average_used_keys:<20}{(num_key_average_used_keys/num_key_average_total_generation_keys) * 100:<4.2f}%")
    # print(f"{'QBER + Num keys':<20}{combination_average_reward:<10}{combination_average_session_blocking:<20}{combination_average_total_generation_keys:<25}{combination_average_used_keys:<20}{(combination_average_used_keys/combination_average_total_generation_keys) * 100:<4.2f}%")

