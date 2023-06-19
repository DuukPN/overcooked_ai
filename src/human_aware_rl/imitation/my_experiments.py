import os
import sys

from behavior_cloning_tf2 import (
    get_bc_params,
    train_bc_model,
    BehaviorCloningPolicy,
    _get_base_ae,
)
from human_aware_rl.rllib.utils import get_base_ae
from human_aware_rl.static import (
    CLEAN_2019_HUMAN_DATA_TEST,
    CLEAN_2019_HUMAN_DATA_TRAIN,
    CLEAN_2019_HUMAN_DATA_ALL,
    HUMAN_DATA_DIR,
)
import human_aware_rl.rllib.rllib as rllib
import numpy as np
from overcooked_ai_py.agents.agent import (
    Agent,
    AgentPair,
    RandomAgent,
)
from human_aware_rl.imitation.scripted_agent import DummyAI

current_file_dir = os.path.dirname(os.path.abspath(__file__))
bc_dir = os.path.join(current_file_dir, "bc_runs")
bc_dir_bc = os.path.join(bc_dir, "bc")
bc_dir_hproxy = os.path.join(bc_dir, "hproxy")
ppo_dir = os.path.join(current_file_dir, "..", "ppo")


class BehaviorCloningAgent(Agent):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def action(self, state):
        actions, states, info = self.policy.compute_actions([state])
        return actions[0], {"action_logits": info["action_dist_inputs"][0]}

    def actions(self, states, agent_indices):
        return [self.action(state) for state in states]


def evaluate_bc_model(name, model_1_dir, model_2_dir, bc_params, verbose=True):
    """
    Creates an AgentPair object containing two instances of BC Agents, whose policies are specified by `model`. Runs
    a rollout using AgentEvaluator class in an environment specified by bc_params

    Arguments

        - model (tf.keras.Model)        A function that maps featurized overcooked states to action logits
        - bc_params (dict)              Specifies the environemnt in which to evaluate the agent (i.e. layout, reward_shaping_param)
                                            as well as the configuration for the rollout (rollout_length)

    Returns

        - reward (int)                  Total sparse reward achieved by AgentPair during rollout
    """
    evaluation_params = bc_params["evaluation_params"]
    mdp_params = bc_params["mdp_params"]

    # Get reference to state encoding function used by bc agents, with compatible signature
    base_ae = _get_base_ae(bc_params)
    base_env = base_ae.env

    def featurize_fn(state):
        return base_env.featurize_state_mdp(state)

    # Wrap Keras models in rllib policies
    agent_0_policy = BehaviorCloningPolicy.from_model_dir(
        model_1_dir, stochastic=True
    )
    agent_1_policy = BehaviorCloningPolicy.from_model_dir(
        model_2_dir, stochastic=True
    )

    # Compute the results of the rollout(s)
    results = rllib.evaluate(
        eval_params=evaluation_params,
        mdp_params=mdp_params,
        outer_shape=None,
        agent_0_policy=agent_0_policy,
        agent_1_policy=agent_1_policy,
        agent_0_featurize_fn=featurize_fn,
        agent_1_featurize_fn=featurize_fn,
        verbose=verbose,
    )
    # Compute the average sparse return obtained in each rollout
    avg_reward = np.mean(results["ep_returns"])
    sd_reward = np.std(results["ep_returns"])

    f = open(os.path.join(bc_dir, "results", f"{name}.txt"), "w")
    f.write(str(results))

    print(f"Successfully completed {name} evaluation")
    print(
        f"Reward: {avg_reward}",
        f"standard deviation: {sd_reward}",
        f"standard error: {sd_reward / np.sqrt(len(results['ep_rewards']))}",
    )
    print(f"Raw data for {name}: \n{str(results)}")

    return results


def fuck_around_and_find_out():
    # layout = "cramped_room"
    # params_to_override = {
    #     # The maps to train on
    #     "layouts": [layout],
    #     # The map to evaluate on
    #     "layout_name": layout,
    #     "data_path": CLEAN_2019_HUMAN_DATA_ALL,
    #     "epochs": 20,
    #     "old_dynamics": True,
    #     "num_games": 100,
    # }
    # bc_params = get_bc_params(**params_to_override)
    # train_bc_model(os.path.join(bc_dir, "train", "my_agent"), bc_params, split=1, verbose=True)

    model = BehaviorCloningPolicy.from_model_dir(os.path.join(bc_dir, "train", "my_agent"))

    if True:
        sys.exit(0)


def train_all_agents(bc_params, layout, terminate=True):
    train_bc_agents(bc_params, layout)
    train_hproxy_agents(bc_params, layout)
    if terminate:
        sys.exit(0)


def train_bc_agents(bc_params, layout):
    # for i in range(5):
    target_dir = os.path.join(bc_dir_bc, f"{layout}")
    if not os.path.isdir(target_dir):
        train_bc_model(target_dir, bc_params, split=1, verbose=True)


def train_hproxy_agents(bc_params, layout):
    # for i in range(5):
    target_dir = os.path.join(bc_dir_hproxy, f"{layout}")
    if not os.path.isdir(target_dir):
        train_bc_model(target_dir, bc_params, split=2, verbose=True)


if __name__ == "__main__":
    # fuck_around_and_find_out()

    # random 3 is counter_circuit
    # random 0 is forced coordination

    # Epoch numbers for each map, in agreement with the original study.
    epoch_dict = {
        "random3": 110,  # counter circuit
        "coordination_ring": 120,
        "cramped_room": 100,
        "random0": 90,  # forced coordination
        "asymmetric_advantages": 120,
    }

    # Path to each PPO agent
    ppo_dict = {
        "random3": os.path.join(ppo_dir, "reproduced_results", "ppo_sp_counter_circuit",
                                "PPO_counter_circuit_o_1order_False_nw=6_vf=0.009920_es=0.200000_en=0.100000_kl=0.299000_0_2023-06-19_12-38-34wevafk1c",
                                "checkpoint_000650"),  # counter circuit
        "coordination_ring": os.path.join(ppo_dir, "reproduced_results", "ppo_sp_coordination_ring",
                                          "PPO_coordination_ring_False_nw=2_vf=0.009330_es=0.200000_en=0.100000_kl=0.156000_0_2023-06-17_14-02-517740nynm",
                                          "checkpoint_000650"),
        "cramped_room": os.path.join(ppo_dir, "reproduced_results", "ppo_sp_cramped_room",
                                     "PPO_cramped_room_False_nw=4_vf=0.009950_es=0.200000_en=0.100000_kl=0.197000_0_2023-06-19_09-49-46x2if1j8l",
                                     "checkpoint_000550"),
        "random0": os.path.join(ppo_dir, "reproduced_results", "ppo_sp_forced_coordination",
                                "PPO_forced_coordination_False_nw=2_vf=0.016000_es=0.200000_en=0.100000_kl=0.310000_0_2023-06-18_17-50-28lxpx5dla",
                                "checkpoint_000650"),  # forced coordination
        "asymmetric_advantages": os.path.join(ppo_dir, "reproduced_results", "ppo_sp_asymmetric_advantages",
                                              "PPO_asymmetric_advantages_False_nw=2_vf=0.022000_es=0.200000_en=0.100000_kl=0.185000_0_2023-06-18_10-32-02hqt2uzy7",
                                              "checkpoint_000650"),
    }

    # for layout in [
    #     "random3",  # counter circuit
    #     "coordination_ring",
    #     "cramped_room",
    #     "random0",  # forced coordination
    #     "asymmetric_advantages",
    # ]:

    layout = sys.argv[1] if len(sys.argv) > 1 else "coordination_ring"

    params_to_override = {
        # The maps to train on
        "layouts": [layout],
        # The map to evaluate on
        "layout_name": layout,
        "data_path": CLEAN_2019_HUMAN_DATA_ALL,
        "epochs": epoch_dict[layout],
        "old_dynamics": True,
        "num_games": 100,
    }
    bc_params = get_bc_params(**params_to_override)

    # train_all_agents(bc_params, layout, False)

    # Create agents
    evaluator = get_base_ae(
        bc_params["mdp_params"],
        {"horizon": bc_params["evaluation_params"]["ep_length"], "num_mdp": 1},
    )

    standard_featurize_fn = evaluator.env.featurize_state_mdp

    hproxy_policy = BehaviorCloningPolicy.from_model_dir(os.path.join(bc_dir, "hproxy", layout))
    hproxy_agent_0 = rllib.RlLibAgent(hproxy_policy, 0, standard_featurize_fn)
    hproxy_agent_1 = rllib.RlLibAgent(hproxy_policy, 1, standard_featurize_fn)

    scripted_agent_0 = DummyAI(0)
    scripted_agent_1 = DummyAI(1)

    ppo_agent_0 = rllib.load_agent(ppo_dict[layout], agent_index=0)
    ppo_agent_1 = rllib.load_agent(ppo_dict[layout], agent_index=1)

    featurize_fns = [standard_featurize_fn]
    bc_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    bc_policy = BehaviorCloningPolicy.from_model_dir(os.path.join(bc_dir, f"bc{bc_idx if bc_idx else ''}", layout))
    bc_agent_0 = rllib.RlLibAgent(bc_policy, 0, featurize_fns[bc_idx])
    bc_agent_1 = rllib.RlLibAgent(bc_policy, 1, featurize_fns[bc_idx])

    print(f"Successfully created all agents\nRunning experiment {bc_idx}")

    result_dir = os.path.join(bc_dir, "results", f"experiment{bc_idx}")
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    tests = ["BC"]
    eval_params = bc_params["evaluation_params"]
    with open(os.path.join(result_dir, f"{layout}_raw.txt"), "w") as raw_file:
        # H_proxy tests
        name = f"{tests[bc_idx]}+H_proxy_0"
        ap = AgentPair(bc_agent_0, hproxy_agent_1)
        results = evaluator.evaluate_agent_pair(
            ap,
            num_games=eval_params["num_games"]
        )
        raw_file.write(f"{name}\n{results['ep_returns']}\n\n")

        name = f"{tests[bc_idx]}+H_proxy_1"
        ap = AgentPair(hproxy_agent_0, bc_agent_1)
        results = evaluator.evaluate_agent_pair(
            ap,
            num_games=eval_params["num_games"]
        )
        raw_file.write(f"{name}\n{results['ep_returns']}\n\n")

        # PPO tests
        name = f"{tests[bc_idx]}+PPO_0"
        ap = AgentPair(bc_agent_0, ppo_agent_1)
        results = evaluator.evaluate_agent_pair(
            ap,
            num_games=eval_params["num_games"]
        )
        raw_file.write(f"{name}\n{results['ep_returns']}\n\n")

        name = f"{tests[bc_idx]}+PPO_1"
        ap = AgentPair(ppo_agent_0, bc_agent_1)
        results = evaluator.evaluate_agent_pair(
            ap,
            num_games=eval_params["num_games"]
        )
        raw_file.write(f"{name}\n{results['ep_returns']}\n\n")

        # Scripted tests
        name = f"{tests[bc_idx]}+Scripted_0"
        ap = AgentPair(bc_agent_0, scripted_agent_1)
        results = evaluator.evaluate_agent_pair(
            ap,
            num_games=eval_params["num_games"]
        )
        raw_file.write(f"{name}\n{results['ep_returns']}\n\n")

        name = f"{tests[bc_idx]}+Scripted_1"
        ap = AgentPair(scripted_agent_0, bc_agent_1)
        results = evaluator.evaluate_agent_pair(
            ap,
            num_games=eval_params["num_games"]
        )
        raw_file.write(f"{name}\n{results['ep_returns']}\n\n")
