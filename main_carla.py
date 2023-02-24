#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 16:46:45 2023

@author: dell
"""

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
        last_layer_dim_co: int = 64,
    ):
        super(CustomNetwork, self).__init__()
        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.latent_dim_co = last_layer_dim_co

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )
        # Cost network
        self.cost_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_co), nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features), self.forward_cost(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)
    
    def forward_cost(self, features: th.Tensor) -> th.Tensor:
        return self.cost_net(features)

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser() 
    parser.add_argument("--env", default = "gymCarla-v0")
    
    args = parser.parse_args()
    env = make_vec_env(args.env)
    
    
    model = PPO(CustomActorCriticPolicy, env, verbose=1, tensorboard_log=("./tsb/ppo_carla_tensorboard/"))
    model.learn(1230000, tb_log_name="carla_test_2_22")
    
    # model = PPO(CustomActorCriticPolicy, env, verbose=1)
    # model.learn(1230000)
    
    model.save("ppo_Carla")
    
    del model # remove to demonstrate saving and loading
    
    model = PPO.load("ppo_Carla")
    
    mean_reward, std_reward, mean_cost, std_cost = evaluate_policy(model,env,n_eval_episodes=1000, deterministic=True)
    print("Mean reward:", mean_reward)
    print("Standard deviation of reward:", std_reward)
    print("Mean cost:", mean_cost)
    print("Standard deviation of cost:", std_cost)
    
    for episode in range(200):
        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, cost, dones, info = env.step(action)
            env.render()
            if dones.any():
                break
        
                
