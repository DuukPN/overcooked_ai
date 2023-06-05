import os
import sys

from behavior_cloning_tf2 import (
    get_bc_params,
    train_bc_model,
    BehaviorCloningPolicy,
    _get_base_ae,
)
from human_aware_rl.static import (
    CLEAN_2019_HUMAN_DATA_TEST,
    CLEAN_2019_HUMAN_DATA_TRAIN,
)
import human_aware_rl.rllib.rllib as rllib
import numpy as np
import threading


current_file_dir = os.path.dirname(os.path.abspath(__file__))
bc_dir = os.path.join(current_file_dir, "bc_runs", "train")


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


if __name__ == "__main__":
    # random 3 is counter_circuit
    # random 0 is forced coordination

    my_agent_dir = os.path.join(bc_dir, "my_agent")
    # if os.path.isdir(bc_dir):
    #     # if this bc agent has been created, we continue to the next layout
    #     continue
    epoch_dict = {
        "random3": 110,  # counter circuit
        "coordination_ring": 120,
        "cramped_room": 100,
        "random0": 90,  # forced coordination
        "asymmetric_advantages": 120,
    }

    # for layout in [
    #     "random3",  # counter circuit
    #     "coordination_ring",
    #     "cramped_room",
    #     "random0",  # forced coordination
    #     "asymmetric_advantages",
    # ]:

    layout = sys.argv[1] if len(sys.argv) > 1 else "cramped_room"

    params_to_override = {
        # The maps to train on
        "layouts": [layout],
        # The map to evaluate on
        "layout_name": layout,
        "data_path": CLEAN_2019_HUMAN_DATA_TRAIN,
        "epochs": epoch_dict[layout],
        "old_dynamics": True,
        "num_games": 100,
    }
    bc_params = get_bc_params(**params_to_override)
    curr_dir_1, curr_dir_2 = \
        os.path.join(bc_dir, f"{layout}_1"), \
        os.path.join(bc_dir, f"{layout}_2")

    if not os.path.isdir(curr_dir_1):
        # threading.Thread(target=train_bc_model, args=(curr_dir_1, bc_params, True)).start()
        train_bc_model(curr_dir_1, bc_params, verbose=True)
    if not os.path.isdir(curr_dir_2):
        # threading.Thread(target=train_bc_model, args=(curr_dir_2, bc_params, True)).start()
        train_bc_model(curr_dir_2, bc_params, verbose=True)

    switched = False
    if len(sys.argv) > 2 and sys.argv[2] == "-s":
        switched = True

    if not switched:
        # threading.Thread(
        #     target=evaluate_bc_model,
        #     args=(f"{layout}_1", curr_dir_1, curr_dir_2, bc_params),
        # ).start()
        results = evaluate_bc_model(f"{layout}_1", curr_dir_1, curr_dir_2, bc_params)
    else:
        # threading.Thread(
        #   target=evaluate_bc_model,
        #   args=(f"{layout}_2", curr_dir_2, curr_dir_1, bc_params),
        # ).start()
        results = evaluate_bc_model(f"{layout}_2", curr_dir_2, curr_dir_1, bc_params)
