import copy

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import topology_conf


class Request:
    def __init__(self):
        self.discard_time = 0
        self.ID = 0


class QuantumEnvironment:
    def __init__(self, topology_type):
        self.G = None
        self.topology_list = {
            'SIMPLE': topology_conf.simple_topo,
            'BUTTERFLY': topology_conf.butterfly_topo,
            'KREONET': topology_conf.kreonet_topo,
            'NSFNET': topology_conf.nsfnet_topo,
            'COST266': topology_conf.cost266_topo
        }
        self.topology_conf = self.topology_list[topology_type]
        self.metric_type = 'qber'   # type: 'simple_shortest', 'weighted_shortest', 'qber', 'num_key', 'combination'
        self.num_seed = 0
        self.max_time_step = 0
        self.training = None

        self.generate_key_size = 0
        self.generate_key_time_slot = 0
        self.consume_key_size = 0
        self.consume_mean = 0
        self.consume_std_dev = 0
        self.num_request = 0
        self.init_qber = None
        self.init_num_channel = None
        self.key_pool_size = 0
        self.Key_pool = None
        self.key_life_time = 0
        self.time_step = 0
        self.source_node = None
        self.target_node = None
        self.service_duration_time = None
        self.service_routing_path = None

        self.session_blocking = 0
        self.total_generation_keys = 0
        self.remaining_keys = 0
        self.used_keys = 0

        self.k = 0
        self.reward = 0
        self.mean_value = 0
        self.std_deviation = 0
        self.count = 0
        self.no_path_count = 0
        self.cumulative_size = 1
        self.cumulative_edge_keys = None
        self.alpha = 0

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
                "num_key": self.topology_conf['num_key'][n],
                "num_channel": self.topology_conf['num_channel'][n]
            } for n in range(len(edges))
        }
        nx.set_edge_attributes(self.G, edges_attribute)
        self.key_pool.update((key, []) for key in self.G.edges)  # Generate key pool
        self.key_generation()  # Reflect the number of keys with qber

    def key_generation(self):
        edges = list(self.G.edges())
        # Update qber and count rate
        # for edge in edges:
        #     # qber scope is 1% ~ 15%
        #     self.G[edge[0]][edge[1]]['qber'] = min(max(round(np.random.binomial(
        #         n=self.G[edge[0]][edge[1]]['count_rate'], p=self.G[edge[0]][edge[1]]['init_qber'] / 100
        #     ) / self.G[edge[0]][edge[1]]['count_rate'] * 100), 1), 15)
        #
        #     if self.time_step != 0:
        #         self.G[edge[0]][edge[1]]['count_rate'] = np.random.randint(1, 5) * 100

        # Generate key with qber
        for edge in edges:
            error_rate = self.G[edge[0]][edge[1]]['qber'] / 100
            ############# Real key rate version #############
            try:
                # generated_keys = round(
                #     self.generate_key_size * max(
                #         1 + error_rate * np.log2(error_rate) + (1 - error_rate) * np.log2(1 - error_rate), 0
                #     )
                # )
                # np.random.seed()
                ######### Apply Pareto distribution #########
                # generated_keys = self.generate_key_size - np.random.pareto(1.5, 1).astype(int)[0] * 100
                # np.random.seed(self.num_seed)
                ######### Apply static generated key #########
                generated_keys = int(np.random.normal(loc=self.generate_key_size, scale=1, size=1))
                # generated_keys = self.generate_key_size

                # print("Gen key: ", generated_keys)
                if generated_keys < 0:
                    generated_keys = 0

                if self.topology_conf['NAME'] == 'BUTTERFLY':
                    if edge[0] == 0 or edge[1] == 5:
                        generated_keys += 10
                self.total_generation_keys += generated_keys

                # Append key life time
                # Simple topo version and else
                if self.topology_conf['NAME'] == 'SIMPLE':
                    self.key_pool[(0, 1)] = [1, 2, 2, 100]
                    self.key_pool[(0, 2)] = [100, 100, 100, 100, 100, 100]
                    self.key_pool[(1, 3)] = [1, 2, 2, 100]
                    self.key_pool[(2, 3)] = [100, 100, 100, 100, 100, 100]
                else:
                    for _ in range(generated_keys):
                        if len(self.key_pool[edge]) + generated_keys > self.key_pool_size:
                            self.key_pool[edge] = self.key_pool[edge][len(self.key_pool[edge]) + generated_keys - self.key_pool_size:]
                        self.key_pool[edge].append(self.key_life_time)

                self.G[edge[0]][edge[1]]['num_key'] = len(self.key_pool[edge])
            except ValueError as e:
                print(error_rate, e)

        # print(self.time_step, [d['num_key'] for u, v, d in self.G.edges(data=True)])
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
        # position = {
        #     0: [2, 12], 1: [4, 14], 2: [4, 10], 3: [8, 14], 4: [8, 10], 5: [10, 12]
        # }
        edge_labels = {}
        pos = nx.spring_layout(self.G)

        # nx.draw(self.G, pos=position, node_color=self.topology_conf['QKD_NODES_COLOR_MAP'], with_labels=True)
        if self.topology_conf['NAME'] == 'KREONET':
            kreonet_pos = {
                0: [2, 10], 1: [3, 9], 2: [3, 11], 3: [9, 13], 4: [11, 12], 5: [10, 11], 6: [4, 8], 7: [6, 7],
                8: [8, 7], 9: [7, 6], 10: [5, 5], 11: [3, 2], 12: [1, 1], 13: [10, 2], 14: [12, 2], 15: [13, 3],
                16: [14, 5], 17: [10, 4]
            }
            labels = {
                0: 'IC', 1: 'SW', 2: 'SO', 3: 'CC', 4: 'GN', 5: 'PC', 6: 'CA', 7: 'SJ',
                8: 'OC', 9: 'DJ', 10: 'JJ', 11: 'GJ', 12: 'JE', 13: 'CW', 14: 'BS', 15: 'US',
                16: 'PH', 17: 'DG'
            }
            nx.draw(self.G, kreonet_pos, with_labels=True, labels=labels)
        else:
            nx.draw(self.G, pos, with_labels=True)
            # for u, v, attr in self.G.edges(data=True):
            #     edge_labels[(u, v)] = "{0}".format(attr['num_key'])
            # nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)
        plt.show()

    def plot_heatmap(self):
        # # Plot the lower triangular part of the heatmap with a red colormap
        # lower_triangular = np.tril(self.node_num_heat)
        # plt.imshow(lower_triangular, cmap='hot_r', interpolation='nearest', alpha=0.7, vmin=0, vmax=1)
        # plt.colorbar(label='Link Strength')
        #
        # # Set the background color to white for the parts that do not appear
        # plt.gca().set_facecolor('white')
        #
        # plt.title('Sophisticated Network with Red Link Strength Heatmap')
        sns.heatmap(self.node_num_heat, annot=False, cmap='coolwarm')
        plt.show()

    def reset(self, seed, max_time_step, training):
        self.num_seed = seed
        np.random.seed(self.num_seed)
        self.max_time_step = max_time_step
        self.training = training

        self.generate_key_time_slot = 15
        self.generate_key_size = 3
        # self.generate_key_size = np.random.pareto(1, 1).astype(int)[0] * 20
        self.init_num_channel = 3
        self.consume_key_size = 1
        self.consume_mean = 1
        self.consume_std_dev = 2
        self.num_request = 0
        self.key_life_time = 20
        self.key_pool_size = 100_000
        self.key_pool = {}
        self.service_duration_time = []
        self.service_routing_path = []

        self.time_step = 0
        self.session_blocking = 0
        self.total_generation_keys = 0
        self.remaining_keys = 0
        self.used_keys = 0
        self.k = 3
        self.reward = 0
        self.alpha = 0.0001

        self.generate_topology()
        self.node_num_heat = np.zeros((len(self.G), len(self.G)))
        self.count = 0
        self.no_path_count = 0
        self.cumulative_size = 5
        self.cumulative_edge_keys = {}
        for edge in self.G.edges:
            self.cumulative_edge_keys[edge] = []
            self.G[edge[0]][edge[1]]['num_channel'] = self.init_num_channel

        # self.calculate_based_lifetime_weight(self.G)
        self.calculate_based_num_key_weight(self.G)

        self.source_node, self.target_node = np.random.choice(np.arange(0, self.topology_conf['NUM_QKD_NODE']), size=2, replace=False)
        # self.source_node, self.target_node = 0, self.topology_conf['NUM_QKD_NODE'] - 1

        state = self.generate_state()
        info = {}
        # self.observation_space = spaces

        return state, info

    def step(self, action):
        state = {}
        info = {}
        done = False
        truncated = False
        next_state = {}
        self.reward = 0
        if self.topology_conf['NAME'] == 'SIMPLE':
            self.source_node, self.target_node = 0, 3
        # else:
        #     if len(action) == 0:
                # self.source_node, self.target_node = random.sample(range(0, self.topology_conf['NUM_QKD_NODE']), 2)
                # self.source_node, self.target_node = np.random.choice(np.arange(0, self.topology_conf['NUM_QKD_NODE']), size=2, replace=False)
                # self.source_node, self.target_node = 0, self.topology_conf['NUM_QKD_NODE'] - 1

        # self.consume_key_size = max(int(np.random.normal(self.consume_mean, self.consume_std_dev)), 1)
        # self.consume_key_size = np.random.pareto(self.consume_mean, 1).astype(int)[0] + 3
        if self.topology_conf['NAME'] == 'SIMPLE':
            if self.time_step == 0:
                self.num_request = 2
            if self.time_step == 1:
                self.num_request = 7
        else:
            # self.num_request = np.random.pareto(4, 1).astype(int)[0] * 5
            # self.num_request = np.random.randint(1, 2, 1)[0]
            self.num_request = 1
        # print("time step: ", self.time_step, "the number of request: ", self.num_request)
        # print("Time step: ", self.time_step, "/ Num request: ", self.num_request, "/ Keys: ", self.key_pool)
        # print(u, v, d['num_key'], d['weight'])

        # Cumulative edge key at cumulative size for weighted average num key
        # for edge in self.G.edges:
        #     if len(self.cumulative_edge_keys[edge]) < 6:
        #         self.cumulative_edge_keys[edge].append(self.G[edge[0]][edge[1]]['num_key'])
        #     else:
        #         self.cumulative_edge_keys[edge].pop(0)
        #         self.cumulative_edge_keys[edge].append(self.G[edge[0]][edge[1]]['num_key'])

        if not self.training:
            for _ in range(self.num_request):
                routing_path = self.find_routing_path()
                # if self.metric_type == 'num_key':
                # print("time step: ", self.time_step, "routing path: ", routing_path)
                #     self.plot_topology()

                if not routing_path:
                    # print("Don't find the routing path")
                    # print("Time step: ", self.time_step, "Blocking reason: ", self.G.edges(data=True))
                    self.session_blocking -= 1
                else:
                    # print("Find the routing path")
                    self.reward += 1
                    # print("src: ", self.source_node, "des: ", self.target_node, "path: ", routing_path)
                    for i in range(len(routing_path) - 1):
                        self.node_num_heat[routing_path[i]][routing_path[i+1]] += 1
                        # self.node_num_heat[routing_path[i+1]][routing_path[i]] += 1
                        self.used_keys += self.consume_key_size

                # for i in self.G.edges:
                #     if i == (0, 1):
                #         # if self.G.edges[i]['num_key'] != len(self.key_pool[i]):
                #         print("time step: ", self.time_step, "edge: ", i, "num_key: ", self.G.edges[i]['num_key'])
                #         print("time step: ", self.time_step, "edge: ", i, "num_key: ", len(self.key_pool[i]), self.key_pool[i])
                #         print()
        else:
            routing_path = action
            check_result = self.check_routing_path(routing_path)
            if not check_result:
                # print("Don't find the routing path")
                # print("Time step: ", self.time_step, "Blocking reason: ", self.G.edges(data=True))
                self.session_blocking -= 1
            else:
                self.service_duration_time.append(5)
                self.service_routing_path.append(routing_path)

                self.apply_routing_path(routing_path)
                # print("Find the routing path")
                self.reward = 1
                for i in range(len(routing_path) - 1):
                    self.node_num_heat[routing_path[i]][routing_path[i + 1]] += 1
                    self.node_num_heat[routing_path[i + 1]][routing_path[i]] += 1
                    self.used_keys += self.consume_key_size

        for u, v, attr in self.G.edges(data=True):
            self.remaining_keys += attr['num_key']
            self.key_pool[(u, v)] = [life - 1 for life in self.key_pool[(u, v)]]
            self.key_pool[(u, v)] = [life for life in self.key_pool[(u, v)] if life >= 1]
            self.G.edges[(u, v)]['num_key'] = len(self.key_pool[(u, v)])

        if self.time_step != 0 and self.time_step % self.generate_key_time_slot == 0:
            self.key_generation()
            # print(self.G.edges(data=True))
        # print(self.metric_type, self.G.edges(data=True))

        self.calculate_based_lifetime_weight(self.G)

        # self.source_node, self.target_node = 0, self.topology_conf['NUM_QKD_NODE'] - 1
        self.source_node, self.target_node = np.random.choice(np.arange(0, self.topology_conf['NUM_QKD_NODE']), size=2, replace=False)

        state = self.generate_state()
        next_state = state

        self.time_step += 1

        info = {
            'session_blocking': self.session_blocking,
            'total_generation_keys': self.total_generation_keys,
            'remaining_keys': self.remaining_keys,
            'used_keys': self.used_keys,
            'graph': self.G
        }

        # Check environment | reflect action | reduction resource
        # print(self.time_step, action)
        # print(self.max_time_step, self.reward)
        # self.plot_topology()

        if self.max_time_step == self.time_step:
            done = True

        return next_state, self.reward, done, truncated, info

    def generate_state(self):
        state = {}
        # Configurate state
        adj_matrix_np = nx.to_numpy_array(self.G, weight=None)
        # Normalization adj matrix
        adj_min, adj_max = adj_matrix_np.min(), adj_matrix_np.max()
        adj_matrix_np = (adj_matrix_np - adj_min) / (adj_max - adj_min)

        # weight_matrix_np = nx.to_numpy_array(self.G, weight='weight')

        num_key_matrix_np = nx.to_numpy_array(self.G, weight='num_key')
        # Normalization weight matrix
        num_key_min, num_key_max = num_key_matrix_np.min(), num_key_matrix_np.max()
        num_key_matrix_np = (num_key_matrix_np - num_key_min) / (num_key_max - num_key_min)

        state['obs'] = np.stack([adj_matrix_np, num_key_matrix_np], axis=0)  # staked (2, H, W)
        state['obs'] = state['obs'][np.newaxis, :]  # shape convert (1, 2, H, W)

        state['paths'], state['valid_mask'] = self.find_k_shortest_path() # shape (1, 3)
        # 고정 길이
        fixed_len = 32
        # padding or trimming
        state['padding_paths'] = [p[:fixed_len] + [0] * max(0, fixed_len - len(p)) for p in state['paths']]
        # paths_info = []
        # paths_index = self.k * 2
        # start_index = 0
        # for path in state['paths']:
        #     paths_info.append(len(path))
        #     paths_info.append(paths_index + start_index)
        #     start_index += len(path)
        # flattened_paths = [node for path in state['paths'] for node in path]
        # paths_info.extend(flattened_paths)
        # paths_info = paths_info + [0] * (128 - len(paths_info))
        #
        # # Normalization flat_paths
        # paths_info = np.array(paths_info)
        # # paths_min, paths_max = paths_info.min(), paths_info.max()
        # # paths_info = (paths_info - paths_min) / (paths_max - paths_min)
        # state['flat_paths'] = paths_info

        # Transform np.array
        state['obs'] = np.array(state['obs'])
        state['padding_paths'] = np.array(state['padding_paths'])
        state['padding_paths'] = state['padding_paths'][np.newaxis, :]
        state['valid_mask'] = np.array(state['valid_mask'])

        return state

    def temp_shrotest_path(self, weight):
        copied_G = copy.deepcopy(self.G)
        subnet = nx.subgraph_view(
            copied_G,
            filter_edge=lambda node_1_id, node_2_id: \
                True if copied_G.edges[(node_1_id, node_2_id)]['num_key'] >= self.consume_key_size else False
        )
        if len(subnet.edges) == 0 or not nx.has_path(subnet, source=self.source_node, target=self.target_node):
            return []

        for edge in subnet.edges:
            subnet[edge[0]][edge[1]]['weight'] = 1 / subnet[edge[0]][edge[1]]['num_key']
        routing_path = nx.astar_path(subnet, self.source_node, self.target_node, None, weight)
        # routing_path = nx.shortest_path(subnet, self.source_node, self.target_node)

        return routing_path

    def check_routing_path(self, routing_path):
        result = True
        if len(routing_path) == 0:
            return False
        copied_G = copy.deepcopy(self.G)
        subnet = nx.subgraph_view(
            copied_G,
            filter_edge=lambda node_1_id, node_2_id: \
                True if copied_G.edges[(node_1_id, node_2_id)]['num_key'] >= self.consume_key_size else False
        )
        # subnet = nx.subgraph_view(
        #     copied_G,
        #     filter_edge=lambda node_1_id, node_2_id: \
        #         True if copied_G.edges[(node_1_id, node_2_id)]['num_key'] >= self.consume_key_size and
        #                 copied_G.edges[(node_1_id, node_2_id)]['num_channel'] > 0 else False
        # )
        if len(subnet.edges) == 0 or not nx.has_path(subnet, source=self.source_node, target=self.target_node):
            return False

        for i in range(len(routing_path) - 1):
            if routing_path[i] < routing_path[i+1]:
                if self.G[routing_path[i]][routing_path[i+1]]['num_key'] < self.consume_key_size:
                    result = False
            else:
                if self.G[routing_path[i+1]][routing_path[i]]['num_key'] < self.consume_key_size:
                    result = False

        return result

    def calculate_based_lifetime_weight(self, net):
        for edge in net.edges:
            life_time_weight = []
            for key_life in self.key_pool[edge]:
                if key_life <= self.key_life_time * 0.1:
                    life_time_weight.append(1000)
                elif key_life <= self.key_life_time * 0.5:
                    life_time_weight.append(100)
                else:
                    life_time_weight.append(1)
            if sum(life_time_weight) != 0:
                # net[edge[0]][edge[1]]['weight'] = len(life_time_weight) / sum(life_time_weight)
                net[edge[0]][edge[1]]['weight'] = (len(self.key_pool[edge]) * 1) / sum(self.key_pool[edge])
            elif sum(life_time_weight) == 0:
                net[edge[0]][edge[1]]['weight'] = 0.0

    def calculate_based_num_key_weight(self, net):
        for edge in net.edges:
            life_time_weight = []
            for key_life in self.key_pool[edge]:
                life_time_weight.append(1)
            if sum(life_time_weight) != 0:
                # net[edge[0]][edge[1]]['weight'] = len(life_time_weight) / sum(life_time_weight)
                net[edge[0]][edge[1]]['weight'] = 1 / sum(self.key_pool[edge])
            elif sum(life_time_weight) == 0:
                net[edge[0]][edge[1]]['weight'] = 0.0

    def find_k_shortest_path(self):
        copied_G = copy.deepcopy(self.G)
        subnet = nx.subgraph_view(
            copied_G,
            filter_edge=lambda node_1_id, node_2_id: \
                True if copied_G.edges[(node_1_id, node_2_id)]['num_key'] >= self.consume_key_size else False
        )
        # subnet = nx.subgraph_view(
        #     copied_G,
        #     filter_edge=lambda node_1_id, node_2_id: \
        #         True if copied_G.edges[(node_1_id, node_2_id)]['num_key'] >= self.consume_key_size and
        #                 copied_G.edges[(node_1_id, node_2_id)]['num_channel'] > 0 else False
        # )
        paths = []
        paths_mask = []
        if len(subnet.edges) == 0 or not nx.has_path(subnet, source=self.source_node, target=self.target_node):
            for _ in range(self.k):
                paths.append([])
                paths_mask.append(0)
            paths_mask.append(1)
            return paths, paths_mask

        # routing_path = nx.shortest_path(subnet, 0, 5)
        for edge in subnet.edges:
            subnet[edge[0]][edge[1]]['weight'] = 1 / subnet[edge[0]][edge[1]]['num_key']
        # paths = list(nx.shortest_simple_paths(subnet, self.source_node, self.target_node, 'weight'))
        # paths = list(nx.all_shortest_paths(subnet, self.source_node, self.target_node, 'weight'))
        try:
            # 다양한 경로를 순차적으로 얻음 (가중치 기준)
            # all_paths_iter = nx.shortest_simple_paths(subnet, self.source_node, self.target_node, weight='weight')
            all_paths_iter = nx.shortest_simple_paths(subnet, self.source_node, self.target_node)

            for path in all_paths_iter:
                paths.append(path)
                paths_mask.append(1)
                if len(paths) >= self.k:
                    break
        except nx.NetworkXNoPath:
            pass

            # 부족한 부분은 padding
        while len(paths) < self.k:
            paths.append([])
            paths_mask.append(0)

        if len(paths) == self.k:
            paths_mask.append(0)

        return paths, paths_mask

    def find_routing_path(self):
        accumulate_qber = []
        accumulate_num_key = []
        accumulate_count_rate = []
        loop_nodes = []
        routing_path = []
        routing_path.append(self.source_node)

        # Using weighted shortest path
        # routing_path = nx.shortest_path(self.G, source=0, target=5, weight='num_key')

        if self.metric_type == 'simple_shortest':
            routing_path = nx.shortest_path(self.G, self.source_node, self.target_node)
            for i in range(len(routing_path) - 1):
                if self.G[routing_path[i]][routing_path[i+1]]['num_key'] < self.consume_key_size:
                    return []
            # copied_G = copy.deepcopy(self.G)
            # subnet = nx.subgraph_view(
            #     copied_G,
            #     filter_edge=lambda node_1_id, node_2_id: \
            #         True if copied_G.edges[(node_1_id, node_2_id)]['num_key'] >= self.consume_key_size else False
            # )
            # if len(subnet.edges) == 0 or not nx.has_path(subnet, source=self.source_node, target=self.target_node):
            #     return []
            #
            # # routing_path = nx.shortest_path(subnet, 0, 5)
            # for edge in subnet.edges:
            #     subnet[edge[0]][edge[1]]['weight'] = 1 / subnet[edge[0]][edge[1]]['num_key']
            # routing_path = nx.astar_path(subnet, self.source_node, self.target_node, None, 'weight')

        if self.metric_type == 'weighted_shortest':
            copied_G = copy.deepcopy(self.G)
            subnet = nx.subgraph_view(
                copied_G,
                filter_edge=lambda node_1_id, node_2_id: \
                    True if copied_G.edges[(node_1_id, node_2_id)]['num_key'] >= self.consume_key_size else False
            )
            if len(subnet.edges) == 0 or not nx.has_path(subnet, source=self.source_node, target=self.target_node):
                return []

            # routing_path = nx.shortest_path(subnet, 0, 5)
            for edge in subnet.edges:
                subnet[edge[0]][edge[1]]['weight'] = 1 / subnet[edge[0]][edge[1]]['num_key']
            routing_path = nx.shortest_path(subnet, self.source_node, self.target_node, 'weight')

        if self.metric_type == 'weighted_life_shortest':
            copied_G = copy.deepcopy(self.G)
            subnet = nx.subgraph_view(
                copied_G,
                filter_edge=lambda node_1_id, node_2_id: \
                    True if copied_G.edges[(node_1_id, node_2_id)]['num_key'] >= self.consume_key_size else False
            )
            # Don't find the path
            if len(subnet.edges) == 0:
                self.no_path_count += 1
                # print("Don't find path: ", self.no_path_count, "({0}, {1})".format(self.source_node, self.target_node))
                return []
            if not nx.has_path(subnet, source=self.source_node, target=self.target_node):
                self.no_path_count += 1
                # print("Don't find path: ", self.no_path_count, "({0}, {1})".format(self.source_node, self.target_node))
                return []

            # routing_path = nx.shortest_path(subnet, 0, 5)
            self.calculate_based_lifetime_weight(subnet)

            routing_path = nx.shortest_path(subnet, self.source_node, self.target_node, 'weight')

        # Using QBER
        if self.metric_type == 'qber':
            # shortest_routing_path = self.temp_shrotest_path(self.source_node, self.target_node, 'weight')
            while self.source_node != self.target_node:
                neighbor_nodes = [node for node in self.G.neighbors(self.source_node) if
                                  self.G[self.source_node][node]['num_key'] > 0 and
                                  self.G[self.source_node][node]['num_key'] >= self.consume_key_size]
                neighbor_nodes = [node for node in neighbor_nodes if node not in routing_path]  # check in routing path
                neighbor_nodes = [node for node in neighbor_nodes if node not in loop_nodes]  # check the loop

                if len(neighbor_nodes) == 0 and len(accumulate_count_rate) > 0:
                    loop_nodes.append(self.source_node)
                    routing_path.pop()
                    accumulate_qber.pop()
                    accumulate_count_rate.pop()
                    self.source_node = routing_path[-1]
                    continue

                elif len(neighbor_nodes) == 0 and len(accumulate_count_rate) == 0:
                    return []

                selected_node = self.select_next_node(neighbor_nodes, accumulate_qber, accumulate_num_key, accumulate_count_rate)
                routing_path.append(selected_node)
                self.source_node = selected_node

            # if len(shortest_routing_path) != 0 and len(shortest_routing_path) * 2 <= len(routing_path):
            #     routing_path = shortest_routing_path

        # Using Num_key
        if self.metric_type == 'num_key':
            # shortest_routing_path = self.temp_shrotest_path(self.source_node, self.target_node, 'weight')
            while self.source_node != self.target_node:
                neighbor_nodes = [node for node in self.G.neighbors(self.source_node) if
                                  self.G[self.source_node][node]['num_key'] > 0 and
                                  self.G[self.source_node][node]['num_key'] >= self.consume_key_size]
                neighbor_nodes = [node for node in neighbor_nodes if node not in routing_path]  # check in routing path
                neighbor_nodes = [node for node in neighbor_nodes if node not in loop_nodes]  # check the loop

                if len(neighbor_nodes) == 0 and len(accumulate_count_rate) > 0:
                    loop_nodes.append(self.source_node)
                    routing_path.pop()
                    accumulate_num_key.pop()
                    accumulate_count_rate.pop()
                    self.source_node = routing_path[-1]
                    continue

                elif len(neighbor_nodes) == 0 and len(accumulate_count_rate) == 0:
                    return []

                selected_node = self.select_next_node(self.source_node, neighbor_nodes, accumulate_qber, accumulate_num_key, accumulate_count_rate)
                routing_path.append(selected_node)
                self.source_node = selected_node

            # if len(shortest_routing_path) != 0 and len(shortest_routing_path) * 2 <= len(routing_path):
            #     routing_path = shortest_routing_path

        # Using QBER + Num_key
        if self.metric_type == 'combination':
            # shortest_routing_path = self.temp_shrotest_path(self.source_node, self.target_node, 'weight')
            while self.source_node != self.target_node:
                neighbor_nodes = [node for node in self.G.neighbors(self.source_node) if
                                  self.G[self.source_node][node]['num_key'] > 0 and
                                  self.G[self.source_node][node]['num_key'] >= self.consume_key_size]
                neighbor_nodes = [node for node in neighbor_nodes if node not in routing_path]  # check in routing path
                neighbor_nodes = [node for node in neighbor_nodes if node not in loop_nodes]  # check the loop

                if len(neighbor_nodes) == 0 and len(accumulate_count_rate) > 0:
                    loop_nodes.append(self.source_node)
                    routing_path.pop()
                    accumulate_qber.pop()
                    accumulate_num_key.pop()
                    accumulate_count_rate.pop()
                    self.source_node = routing_path[-1]
                    continue

                elif len(neighbor_nodes) == 0 and len(accumulate_count_rate) == 0:
                    return []

                selected_node = self.select_next_node(
                    self.source_node, neighbor_nodes,
                    accumulate_qber, accumulate_num_key, accumulate_count_rate
                )
                routing_path.append(selected_node)
                self.source_node = selected_node

            # print("!!!!!!!!!!", len(routing_path), routing_path, len(shortest_routing_path), shortest_routing_path)
            # if len(shortest_routing_path) != 0 and len(shortest_routing_path) * 2 <= len(routing_path):
            #     routing_path = shortest_routing_path
            #     self.count += 1
            # print(self.count)

        # if self.metric_type == 'combination':
        #     print("Final routing path: ", routing_path)
        # Consumed quantum key

        self.apply_routing_path(routing_path)

        return routing_path

    def apply_routing_path(self, routing_path):
        for i in range(len(routing_path) - 1):
            sorted_key = tuple(sorted((routing_path[i], routing_path[i + 1])))
            try:
                self.G[sorted_key[0]][sorted_key[1]]['num_key'] -= self.consume_key_size
                self.G[sorted_key[0]][sorted_key[1]]['num_channel'] -= 1
                self.key_pool[sorted_key] = self.key_pool[sorted_key][self.consume_key_size:]
            except:
                print(routing_path)

    def select_next_node(self, neighbor_nodes, accumulate_qber, accumulate_num_key, accumulate_count_rate):
        current_edges = list(self.G.edges(self.source_node))
        for edge in current_edges:
            self.G[edge[0]][edge[1]]['weight'] = self.calculate_weight(
                edge, accumulate_qber, accumulate_num_key, accumulate_count_rate,
                self.G[edge[0]][edge[1]]['qber'],
                self.G[edge[0]][edge[1]]['num_key'],
                self.G[edge[0]][edge[1]]['count_rate']
            )

        min_weight_neighbor = min(neighbor_nodes,
                                  key=lambda neighbor: self.G[self.source_node][neighbor].get('weight', float('inf'))
                                  )
        accumulate_qber.append(self.G[self.source_node][min_weight_neighbor]['qber'])
        accumulate_num_key.append(self.G[self.source_node][min_weight_neighbor]['num_key'])
        accumulate_count_rate.append(self.G[self.source_node][min_weight_neighbor]['count_rate'])

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
                current_num_key = 10_000
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
    env = QuantumEnvironment(topology_type='NSFNET')
    max_time_step = 100    # 1_000
    num_simulation = 1
    seed = 0
    action = []

    weighted_shortest_reward, shortest_reward, qber_reward, num_key_reward, combination_reward = 0, 0, 0, 0, 0
    weighted_shortest_average_reward, shortest_average_reward, qber_average_reward, num_key_average_reward, combination_average_reward = 0, 0, 0, 0, 0
    weighted_shortest_average_session_blocking, shortest_average_session_blocking, qber_average_session_blocking, num_key_average_session_blocking, combination_average_session_blocking = 0, 0, 0, 0, 0
    weighted_shortest_average_total_generation_keys, shortest_average_total_generation_keys, qber_average_total_generation_keys, num_key_average_total_generation_keys, combination_average_total_generation_keys = 0, 0, 0, 0, 0
    weighted_shortest_average_remaining_keys, shortest_average_remaining_keys, qber_average_remaining_keys, num_key_average_remaining_keys, combination_average_remaining_keys = 0, 0, 0, 0, 0
    weighted_shortest_average_used_keys, shortest_average_used_keys, qber_average_used_keys, num_key_average_used_keys, combination_average_used_keys = 0, 0, 0, 0, 0

    # Shortest path simulation
    env.metric_type = 'simple_shortest'
    # env.plot_topology()
    for _ in range(num_simulation):
        env.reset(seed=seed, max_time_step=max_time_step, training=False)
        for _ in range(max_time_step):
            _, shortest_reward, _, _, info = env.step(action)
        shortest_average_reward += shortest_reward
        shortest_average_session_blocking += info['session_blocking']
        shortest_average_total_generation_keys += info['total_generation_keys']
        shortest_average_remaining_keys += info['remaining_keys']
        shortest_average_used_keys += info['used_keys']
    shortest_average_reward /= num_simulation
    shortest_average_session_blocking /= num_simulation
    shortest_average_total_generation_keys /= num_simulation
    shortest_average_remaining_keys /= num_simulation
    shortest_average_used_keys /= num_simulation
    # env.plot_topology()
    # env.plot_heatmap()

    # Weighted shortest path simulation
    env.metric_type = 'weighted_shortest'
    env.plot_topology()
    for _ in range(num_simulation):
        s, _ = env.reset(seed=seed, max_time_step=max_time_step, training=False)
        for _ in range(max_time_step):
            _, weighted_shortest_reward, _, _, info = env.step(action)
        weighted_shortest_average_reward += weighted_shortest_reward
        weighted_shortest_average_session_blocking += info['session_blocking']
        weighted_shortest_average_total_generation_keys += info['total_generation_keys']
        weighted_shortest_average_remaining_keys += info['remaining_keys']
        weighted_shortest_average_used_keys += info['used_keys']
    weighted_shortest_average_reward /= num_simulation
    weighted_shortest_average_session_blocking /= num_simulation
    weighted_shortest_average_total_generation_keys /= num_simulation
    weighted_shortest_average_remaining_keys /= num_simulation
    weighted_shortest_average_used_keys /= num_simulation
    # env.plot_topology()
    # env.plot_heatmap()

    # QBER simulation
    env.metric_type = 'weighted_life_shortest'
    # env.plot_topology()
    for _ in range(num_simulation):
        s, _ = env.reset(seed=seed, max_time_step=max_time_step, training=False)
        for _ in range(max_time_step):
            _, qber_reward, _, _, info = env.step(action)
        qber_average_reward += qber_reward
        qber_average_session_blocking += info['session_blocking']
        qber_average_total_generation_keys += info['total_generation_keys']
        qber_average_remaining_keys += info['remaining_keys']
        qber_average_used_keys += info['used_keys']
    qber_average_reward /= num_simulation
    qber_average_session_blocking /= num_simulation
    qber_average_total_generation_keys /= num_simulation
    qber_average_remaining_keys /= num_simulation
    qber_average_used_keys /= num_simulation
    # # env.plot_topology()
    # env.plot_heatmap()
    #
    # # Num keys simulation
    # env.metric_type = 'num_key'
    # env.reset(seed=seed, max_time_step=max_time_step, training=False)
    # for _ in range(num_simulation):
    #     env.reset(seed=seed, max_time_step=max_time_step, training=False)
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
    # env.reset(seed=seed, max_time_step=max_time_step, training=False)
    # for _ in range(num_simulation):
    #     env.reset(seed=seed, max_time_step=max_time_step, training=False)
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
    print(f"{'life-aware_shortest':<20}{qber_average_reward:<10}{qber_average_session_blocking:<20}{qber_average_total_generation_keys:<25}{qber_average_used_keys:<20}{(qber_average_used_keys/qber_average_total_generation_keys) * 100:<4.2f}%")
    # print(f"{'Num keys':<20}{num_key_average_reward:<10}{num_key_average_session_blocking:<20}{num_key_average_total_generation_keys:<25}{num_key_average_used_keys:<20}{(num_key_average_used_keys/num_key_average_total_generation_keys) * 100:<4.2f}%")
    # print(f"{'QBER + Num keys':<20}{combination_average_reward:<10}{combination_average_session_blocking:<20}{combination_average_total_generation_keys:<25}{combination_average_used_keys:<20}{(combination_average_used_keys/combination_average_total_generation_keys) * 100:<4.2f}%")

