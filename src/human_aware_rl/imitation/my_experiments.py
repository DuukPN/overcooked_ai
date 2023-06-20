import os
import sys
from collections import Counter

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
from overcooked_ai_py.mdp.actions import Direction
from overcooked_ai_py.utils import pos_distance
from human_aware_rl.imitation.featurization import create_featurize_fn

current_file_dir = os.path.dirname(os.path.abspath(__file__))
bc_dir = os.path.join(current_file_dir, "bc_runs")
bc_dir_bc = os.path.join(bc_dir, "bc")
bc_dir_hproxy = os.path.join(bc_dir, "hproxy")
ppo_dir = os.path.join(current_file_dir, "..", "ppo")


class BehaviorCloningAgent(Agent):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def action(self, states, actions):
        state = states[-1]
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

    def featurize_fn(states, actions):
        return base_env.featurize_state_mdp(states[-1])

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

    # model = BehaviorCloningPolicy.from_model_dir(os.path.join(bc_dir, "train", "my_agent"))

    import time
    time.sleep(4000)


def train_all_agents(bc_params, layout, bc_idx, standard_featurize_fn, featurize_fn, terminate=True):
    bc_params["featurize_fn"] = featurize_fn
    train_bc_agents(bc_params, layout, bc_idx)
    bc_params["featurize_fn"] = standard_featurize_fn
    train_hproxy_agents(bc_params, layout)
    if terminate:
        sys.exit(0)


def train_bc_agents(bc_params, layout, bc_idx):
    # for i in range(5):
    target_dir = os.path.join(bc_dir, f"bc_{bc_idx}", f"{layout}")
    if not os.path.isdir(target_dir):
        train_bc_model(target_dir, bc_params, split=1, verbose=True)


def train_hproxy_agents(bc_params, layout):
    # for i in range(5):
    target_dir = os.path.join(bc_dir_hproxy, f"{layout}")
    if not os.path.isdir(target_dir):
        train_bc_model(target_dir, bc_params, split=2, verbose=True)


if __name__ == "__main__":
    completed_experiments = {0}
    fuck_around_and_find_out()

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
                                "PPO_counter_circuit",
                                "checkpoint_000650"),  # counter circuit
        "coordination_ring": os.path.join(ppo_dir, "reproduced_results", "ppo_sp_coordination_ring",
                                          "PPO_coordination_ring",
                                          "checkpoint_000650"),
        "cramped_room": os.path.join(ppo_dir, "reproduced_results", "ppo_sp_cramped_room",
                                     "PPO_cramped_room",
                                     "checkpoint_000550"),
        "random0": os.path.join(ppo_dir, "reproduced_results", "ppo_sp_forced_coordination",
                                "PPO_forced_coordination",
                                "checkpoint_000650"),  # forced coordination
        "asymmetric_advantages": os.path.join(ppo_dir, "reproduced_results", "ppo_sp_asymmetric_advantages",
                                              "PPO_asymmetric_advantages",
                                              "checkpoint_000650"),
    }

    # Process command line arguments
    layout = sys.argv[1] if len(sys.argv) > 1 else "coordination_ring"
    bc_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    part = sys.argv[3] if len(sys.argv) > 3 else "hproxy"

    if bc_idx in completed_experiments:
        print(f"Experiment {bc_idx} was already completed. To protect previous data, this experiment will not be repeated.")
        sys.exit(0)

    # for layout in [
    #     "random3",  # counter circuit
    #     "coordination_ring",
    #     "cramped_room",
    #     "random0",  # forced coordination
    #     "asymmetric_advantages",
    # ]:

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
    shapes = [(96,), (50,)]
    bc_params["observation_shape"] = shapes[bc_idx]
    evaluator = get_base_ae(
        bc_params["mdp_params"],
        {"horizon": bc_params["evaluation_params"]["ep_length"], "num_mdp": 1},
    )
    mdp, mlam = evaluator.env.mdp, evaluator.env.mlam

    standard_featurize_fn = create_featurize_fn(0, mdp, mlam)
    current_featurize_fn = create_featurize_fn(bc_idx, mdp, mlam)

    #     train_all_agents(bc_params, layout, bc_idx, standard_featurize_fn, current_featurize_fn, False)
    # else:
    #     sys.exit(0)

    # Create agents

    if part == "hproxy":
        hproxy_policy = BehaviorCloningPolicy.from_model_dir(os.path.join(bc_dir, "hproxy", layout))
        hproxy_agent_0 = rllib.RlLibAgent(hproxy_policy, 0, standard_featurize_fn)
        hproxy_agent_1 = rllib.RlLibAgent(hproxy_policy, 1, standard_featurize_fn)

    if part == "script":
        scripted_agent_0 = DummyAI(0, layout)
        scripted_agent_1 = DummyAI(1, layout)

    if part == "ppo":
        ppo_agent_0 = rllib.load_agent(ppo_dict[layout], agent_index=0)
        ppo_agent_1 = rllib.load_agent(ppo_dict[layout], agent_index=1)

    bc_policy = BehaviorCloningPolicy.from_model_dir(os.path.join(bc_dir, f"bc_{bc_idx}" if bc_idx else 'bc', layout))
    bc_agent_0 = rllib.RlLibAgent(bc_policy, 0, current_featurize_fn)
    bc_agent_1 = rllib.RlLibAgent(bc_policy, 1, current_featurize_fn)

    print(f"Successfully created all agents\nRunning experiment {bc_idx}")

    result_dir = os.path.join(bc_dir, "results", f"experiment{bc_idx}")
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    tests = ["BC", "BC1", "BC2", "BC3", "BC4", "BC5"]
    eval_params = bc_params["evaluation_params"]

    if part == "hproxy":
        with open(os.path.join(result_dir, f"{layout}_hproxy_raw.txt"), "w") as raw_file:
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

    if part == "ppo":
        with open(os.path.join(result_dir, f"{layout}_ppo_raw.txt"), "w") as raw_file:
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

    if part == "script":
        with open(os.path.join(result_dir, f"{layout}_script_raw.txt"), "w") as raw_file:
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
