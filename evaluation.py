import torch
import numpy as np
from training_PPO import PPO
from environment import QuantumEnvironment

def evaluate(model_path, num_episodes=50, max_time_step=100, topology='NSFNET'):
    # 1) 환경과 모델 초기화
    env = QuantumEnvironment(topology_type=topology)
    model = PPO()
    # 2) 저장된 파라미터 로드
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    rewards = []
    for episode in range(num_episodes):
        # seed를 episode index로 주면 반복 실행 시 재현 가능
        state, _ = env.reset(episode, max_time_step, True)
        done = False
        ep_reward = 0.0

        while not done:
            # 상태 텐서 생성
            obs = torch.from_numpy(state['obs']).float()
            obs_path = torch.from_numpy(state['padding_paths']).float()
            mask = torch.tensor(state['valid_mask']).float().unsqueeze(0)

            # 3) Greedy 평가: 확률 최대인 액션 선택
            with torch.no_grad():
                probs = model.pi(obs, obs_path, mask=mask, softmax_dim=1)
            action_idx = probs.argmax(dim=1).item()

            # action 인덱스를 실제 경로로 변환
            paths = state['paths']
            a = paths[action_idx] if action_idx < len(paths) else []

            # 4) 환경에 적용
            state, r, done, truncated, info = env.step(a)
            ep_reward += r

        rewards.append(ep_reward)
        print(f"Episode {episode+1:>2} Episode Reward: {ep_reward:.3f}")


    # 결과 출력
    print(f"Evaluation over {num_episodes} episodes")
    print(f"  Average Reward : {np.mean(rewards):.3f}")
    print(f"  Reward Std Dev : {np.std(rewards):.3f}")
    return rewards

if __name__ == '__main__':
    # 모델 파일 경로 (training_PPO.py 에서 저장한 경로와 일치시킬 것)
    MODEL_PATH = "model_save/PPO_cost266_highest_model_final"
    evaluate(MODEL_PATH, num_episodes=100, max_time_step=100)