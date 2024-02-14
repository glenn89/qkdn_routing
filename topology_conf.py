import numpy as np


################### Butterfly topology ###################
mean_value = 5
std_deviation = 4
init_qber = np.random.normal(mean_value, std_deviation, size=18)
init_qber = np.round(np.clip(init_qber, 1, 10))
init_count_rate = np.random.randint(1, 4, size=18) * 100

butterfly_topo = {
    'NAME': "BUTTERFLY",
    'QKD_NODES': [0, 1, 2, 3, 4, 5],
    'QKD_NODES_NAME': {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'},
    'QKD_NODES_COLOR_MAP': ['red', 'green', 'green', 'green', 'green', 'yellow'],
    'QKD_TOPOLOGY': [
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 0],
        [1, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 1, 1],
        [0, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 1, 0]
    ],
    'INIT_QBER': init_qber,
    'QBER': init_qber, # [16, 2, 16, 5, 1, 2, 5, 1, 3, 5, 1, 2, 3, 3, 2, 4, 3, 4]
    'num_key': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'COUNT_RATE': init_count_rate, # [320, 800, 320, 100, 100, 800, 100, 120, 150, 100, 120, 420, 320, 150, 420, 320, 320, 320]
    'NUM_QKD_NODE': 6,
    'NUM_QKD_LINK': 9,
}

################### KREONET topology ###################
mean_value = 5
std_deviation = 4
init_qber = np.random.normal(mean_value, std_deviation, size=34)
init_qber = np.round(np.clip(init_qber, 1, 10))
init_num_key = np.zeros(34)
init_count_rate = np.random.randint(1, 4, size=34) * 100

kreonet_topo = {
    'NAME': "KREONET",
    'QKD_NODES': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    'QKD_NODES_NAME': {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'},
    'QKD_NODES_COLOR_MAP': ['red', 'green', 'green', 'green', 'green', 'yellow'],
    'QKD_TOPOLOGY': [
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Inchon [2, 10]
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Suwon [3, 9]
        [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Seoul [3, 11]
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Chunchen [9, 13]
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Gangneung [11, 12]
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Pyeongchang [10, 11]
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Cheonan [4, 8]
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Sejong [6, 7]
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Ochang [8, 7]
        [0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],  # Daejeon [7, 6]
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Jeonju [5, 5]
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Gwangju [3, 2]
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Jeju [1, 1]
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Changwon [10, 2]
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Busan [12 ,2]
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Ulsan [13, 3]
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Pohang [14, 5]
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Daegu [10, 4]
    ],
    'INIT_QBER': init_qber,
    'QBER': init_qber, # [16, 2, 16, 5, 1, 2, 5, 1, 3, 5, 1, 2, 3, 3, 2, 4, 3, 4]
    'num_key': init_num_key,
    'COUNT_RATE': init_count_rate, # [320, 800, 320, 100, 100, 800, 100, 120, 150, 100, 120, 420, 320, 150, 420, 320, 320, 320]
    'NUM_QKD_NODE': 18,
    'NUM_QKD_LINK': 17,
}

################### NSFNET topology ###################
mean_value = 5
std_deviation = 4
init_qber = np.random.normal(mean_value, std_deviation, size=44)
init_qber = np.round(np.clip(init_qber, 1, 10))
init_num_key = np.zeros(44)
init_count_rate = np.random.randint(1, 4, size=44) * 100

nsfnet_topo = {
    'NAME': "NSFNET",
    'QKD_NODES': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    'QKD_TOPOLOGY': [
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    ],

    'NUM_QKD_NODE': 14,
    'NUM_QKD_LINK': 22,
    'INIT_QBER': init_qber,
    'QBER': init_qber,  # [16, 2, 16, 5, 1, 2, 5, 1, 3, 5, 1, 2, 3, 3, 2, 4, 3, 4]
    'num_key': init_num_key,
    'COUNT_RATE': init_count_rate,  # [320, 800, 320, 100, 100, 800, 100, 120, 150, 100, 120, 420, 320, 150, 420, 320, 320, 320]
}
