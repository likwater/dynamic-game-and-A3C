import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

# 定义网络模型
def build_model(input_shape, output_shape):
    inputs = Input(shape=input_shape)
    x = Dense(128, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(output_shape, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# 定义动态博弈函数
def dynamic_game(state, defense_action, attack_action):
    attack_success_rate = 0.2
    defense_effectiveness = [
        [0.1, 0.4, 0.7],
        [0.2, 0.5, 0.6],
        [0.3, 0.6, 0.8]
    ]
    defense_cost = [0.1, 0.2, 0.3]
    attack_strength = [0.1, 0.3, 0.6]

    attack_success = np.random.rand() < attack_success_rate * attack_strength[attack_action]
    defense_success = np.random.rand() < defense_effectiveness[defense_action][attack_action]

    if attack_success and not defense_success:
        reward = -1.0
    elif not attack_success and defense_success:
        reward = 1.0 - defense_cost[defense_action]
    else:
        reward = -defense_cost[defense_action]

    state_change = defense_effectiveness[defense_action][attack_action] - attack_strength[attack_action]
    next_state = state + state_change * np.ones_like(state)

    return next_state, reward


class A3C:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.gamma = 0.99
        self.entropy_weight = 0.01
        self.lambda_gae = 0.95

    def train(self, states, actions, cumulative_rewards, advantages):
        with tf.GradientTape() as tape:
            logits = self.model(states)
            action_probs = tf.nn.softmax(logits)
            action_log_probs = tf.nn.log_softmax(logits)
            selected_action_log_probs = tf.reduce_sum(action_log_probs * actions, axis=1)

            entropy = -tf.reduce_sum(action_probs * action_log_probs, axis=1)
            loss_entropy = self.entropy_weight * tf.reduce_mean(entropy)

            loss = -tf.reduce_mean(selected_action_log_probs * advantages) + loss_entropy

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def choose_action(self, state):
        logits = self.model(state[np.newaxis, :])
        action_probs = tf.nn.softmax(logits)
        action = np.random.choice(len(action_probs[0]), p=action_probs[0].numpy())
        return action

    def calculate_gae(self, rewards, values, next_value, done_mask):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae_lambda = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - done_mask[t]
            else:
                next_non_terminal = 1.0 - done_mask[t]
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae_lambda = delta + self.gamma * self.lambda_gae * next_non_terminal * last_gae_lambda
            advantages[t] = last_gae_lambda

        return advantages


# 创建模型和优化器
input_shape = (784,)  # 输入为状态向量的维度
output_shape = 3  # 假设有三种防御策略
model = build_model(input_shape, output_shape)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
a3c = A3C(model, optimizer)

# 保存和加载模型的路径
model_weights_path = 'a3c_model.weights.h5'

# # 训练过程
# num_episodes = 1000
# for episode in range(num_episodes):
#     states, actions, rewards, done_mask = [], [], [], []
#     state = np.random.rand(784)  # 初始状态
#     episode_reward = 0
#     for step in range(100):  # 每个episode的最大步数
#         defense_action = a3c.choose_action(state)
#         attack_action = np.random.choice(output_shape)  # 随机选择一个攻击策略
#         next_state, reward = dynamic_game(state, defense_action, attack_action)
#         episode_reward += reward
#
#         states.append(state)
#         actions.append(defense_action)
#         rewards.append(reward)
#         done_mask.append(0 if step < 99 else 1)
#
#         state = next_state
#
#     states = np.array(states, dtype=np.float32)
#     values = a3c.model(states)
#     next_values = a3c.model(np.array([state], dtype=np.float32))
#
#     advantages = a3c.calculate_gae(rewards, values.numpy().flatten(), next_values.numpy().flatten()[0], np.array(done_mask))
#
#     cumulative_rewards = []
#     cumulative_reward = 0
#     for r in reversed(rewards):
#         cumulative_reward = r + a3c.gamma * cumulative_reward
#         cumulative_rewards.insert(0, cumulative_reward)
#
#     a3c.train(np.array(states, dtype=np.float32),
#               tf.one_hot(actions, output_shape, dtype=tf.float32),
#               np.array(cumulative_rewards, dtype=np.float32),
#               np.array(advantages, dtype=np.float32))
#
#     print(f'Episode {episode + 1}: Reward = {episode_reward}')

# 保存训练完成的模型权重
model.save_weights(model_weights_path)

print("训练完成")

# 测试模型
def judge(num_test_episodes=1000):
    test_rewards = []
    state = np.random.rand(784)  # 初始状态

    # 加载训练完成的模型权重
    model.load_weights(model_weights_path)
    episode_reward = 0
    for episode in range(num_test_episodes):
        defense_action = a3c.choose_action(state)
        attack_action = np.random.choice(output_shape)  # 随机选择攻击策略
        next_state, reward = dynamic_game(state, defense_action, attack_action)
        episode_reward += reward
        state = next_state
        test_rewards.append(episode_reward)
        print(f"Test Episode {episode + 1}: Reward = {episode_reward}")

    return test_rewards

# 测试模型并绘制结果
test_rewards = judge()

# 绘制测试阶段的累计奖励折线图
plt.plot(test_rewards)
plt.xlabel('Test Episode')
plt.ylabel('Accumulated Reward')
plt.title('Test Performance')
plt.show()
