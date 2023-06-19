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


def create_featurize_1(self):
    def featurize_state(overcooked_state, mlam, num_pots=2, **kwargs):
        """
        Encode state with some manually designed features. Works for arbitrary number of players

        Arguments:
            overcooked_state (OvercookedState): state we wish to featurize
            mlam (MediumLevelActionManager): to be used for distance computations necessary for our higher-level feature encodings
            num_pots (int): Encode the state (ingredients, whether cooking or not, etc) of the 'num_pots' closest pots to each player.
                If i < num_pots pots are reachable by player i, then pots [i+1, num_pots] are encoded as all zeros. Changing this
                impacts the shape of the feature encoding

        Returns:
            ordered_features (list[np.Array]): The ith element contains a player-centric featurized view for the ith player

            The encoding for player i is as follows:

                [player_i_features, other_player_features player_i_dist_to_other_players, player_i_position]

                player_{i}_features (length num_pots*10 + 24):
                    pi_orientation: length 4 one-hot-encoding of direction currently facing
                    pi_obj: length 4 one-hot-encoding of object currently being held (all 0s if no object held)
                    pi_wall_{j}: {0, 1} boolean value of whether player i has wall immediately in direction j
                    pi_closest_{onion|tomato|dish|soup|serving|empty_counter}: (dx, dy) where dx = x dist to item, dy = y dist to item. (0, 0) if item is currently held
                    pi_cloest_soup_n_{onions|tomatoes}: int value for number of this ingredient in closest soup
                    pi_closest_pot_{j}_exists: {0, 1} depending on whether jth closest pot found. If 0, then all other pot features are 0. Note: can
                        be 0 even if there are more than j pots on layout, if the pot is not reachable by player i
                    pi_closest_pot_{j}_{is_empty|is_full|is_cooking|is_ready}: {0, 1} depending on boolean value for jth closest pot
                    pi_closest_pot_{j}_{num_onions|num_tomatoes}: int value for number of this ingredient in jth closest pot
                    pi_closest_pot_{j}_cook_time: int value for seconds remaining on soup. -1 if no soup is cooking
                    pi_closest_pot_{j}: (dx, dy) to jth closest pot from player i location

                other_player_features (length (num_players - 1)*(num_pots*10 + 24)):
                    ordered concatenation of player_{j}_features for j != i

                player_i_dist_to_other_players (length (num_players - 1)*2):
                    [player_j.pos - player_i.pos for j != i]

                player_i_position (length 2)
        """

        all_features = {}

        def concat_dicts(a, b):
            return {**a, **b}

        def make_closest_feature(idx, player, name, locations):
            """
            Compute (x, y) deltas to closest feature of type `name`, and save it in the features dict
            """
            feat_dict = {}
            obj = None
            held_obj = player.held_object
            held_obj_name = held_obj.name if held_obj else "none"
            if held_obj_name == name:
                obj = held_obj
                feat_dict["p{}_closest_{}".format(i, name)] = (0, 0)
            else:
                loc, deltas = self.get_deltas_to_closest_location(
                    player, locations, mlam
                )
                if loc and overcooked_state.has_object(loc):
                    obj = overcooked_state.get_object(loc)
                feat_dict["p{}_closest_{}".format(idx, name)] = deltas

            if name == "soup":
                num_onions = num_tomatoes = 0
                if obj:
                    ingredients_cnt = Counter(obj.ingredients)
                    num_onions, num_tomatoes = (
                        ingredients_cnt["onion"],
                        ingredients_cnt["tomato"],
                    )
                feat_dict["p{}_closest_soup_n_onions".format(i)] = [num_onions]
                feat_dict["p{}_closest_soup_n_tomatoes".format(i)] = [
                    num_tomatoes
                ]

            return feat_dict

        def make_pot_feature(idx, player, pot_idx, pot_loc, pot_states):
            """
            Encode pot at pot_loc relative to 'player'
            """
            # Pot doesn't exist
            feat_dict = {}
            if not pot_loc:
                feat_dict["p{}_closest_pot_{}_exists".format(idx, pot_idx)] = [
                    0
                ]
                feat_dict[
                    "p{}_closest_pot_{}_is_empty".format(idx, pot_idx)
                ] = [0]
                feat_dict[
                    "p{}_closest_pot_{}_is_full".format(idx, pot_idx)
                ] = [0]
                feat_dict[
                    "p{}_closest_pot_{}_is_cooking".format(idx, pot_idx)
                ] = [0]
                feat_dict[
                    "p{}_closest_pot_{}_is_ready".format(idx, pot_idx)
                ] = [0]
                feat_dict[
                    "p{}_closest_pot_{}_num_onions".format(idx, pot_idx)
                ] = [0]
                feat_dict[
                    "p{}_closest_pot_{}_num_tomatoes".format(idx, pot_idx)
                ] = [0]
                feat_dict[
                    "p{}_closest_pot_{}_cook_time".format(idx, pot_idx)
                ] = [0]
                feat_dict["p{}_closest_pot_{}".format(idx, pot_idx)] = (0, 0)
                return feat_dict

            # Get position information
            deltas = self.get_deltas_to_location(player, pot_loc)

            # Get pot state info
            is_empty = int(pot_loc in self.get_empty_pots(pot_states))
            is_full = int(pot_loc in self.get_full_pots(pot_states))
            is_cooking = int(pot_loc in self.get_cooking_pots(pot_states))
            is_ready = int(pot_loc in self.get_ready_pots(pot_states))

            # Get soup state info
            num_onions = num_tomatoes = 0
            cook_time_remaining = 0
            if not is_empty:
                soup = overcooked_state.get_object(pot_loc)
                ingredients_cnt = Counter(soup.ingredients)
                num_onions, num_tomatoes = (
                    ingredients_cnt["onion"],
                    ingredients_cnt["tomato"],
                )
                cook_time_remaining = (
                    0 if soup.is_idle else soup.cook_time_remaining
                )

            # Encode pot and soup info
            feat_dict["p{}_closest_pot_{}_exists".format(idx, pot_idx)] = [1]
            feat_dict["p{}_closest_pot_{}_is_empty".format(idx, pot_idx)] = [
                is_empty
            ]
            feat_dict["p{}_closest_pot_{}_is_full".format(idx, pot_idx)] = [
                is_full
            ]
            feat_dict["p{}_closest_pot_{}_is_cooking".format(idx, pot_idx)] = [
                is_cooking
            ]
            feat_dict["p{}_closest_pot_{}_is_ready".format(idx, pot_idx)] = [
                is_ready
            ]
            feat_dict["p{}_closest_pot_{}_num_onions".format(idx, pot_idx)] = [
                num_onions
            ]
            feat_dict[
                "p{}_closest_pot_{}_num_tomatoes".format(idx, pot_idx)
            ] = [num_tomatoes]
            feat_dict["p{}_closest_pot_{}_cook_time".format(idx, pot_idx)] = [
                cook_time_remaining
            ]
            feat_dict["p{}_closest_pot_{}".format(idx, pot_idx)] = deltas

            return feat_dict

        IDX_TO_OBJ = ["onion", "soup", "dish", "tomato"]
        OBJ_TO_IDX = {o_name: idx for idx, o_name in enumerate(IDX_TO_OBJ)}

        counter_objects = self.get_counter_objects_dict(overcooked_state)
        pot_states = self.get_pot_states(overcooked_state)

        for i, player in enumerate(overcooked_state.players):
            # Player info
            orientation_idx = Direction.DIRECTION_TO_INDEX[player.orientation]
            all_features["p{}_orientation".format(i)] = np.eye(4)[
                orientation_idx
            ]
            obj = player.held_object

            if obj is None:
                held_obj_name = "none"
                all_features["p{}_objs".format(i)] = np.zeros(len(IDX_TO_OBJ))
            else:
                held_obj_name = obj.name
                obj_idx = OBJ_TO_IDX[held_obj_name]
                all_features["p{}_objs".format(i)] = np.eye(len(IDX_TO_OBJ))[
                    obj_idx
                ]

            # Closest feature for each object type
            all_features = concat_dicts(
                all_features,
                make_closest_feature(
                    i,
                    player,
                    "onion",
                    self.get_onion_dispenser_locations()
                    + counter_objects["onion"],
                ),
            )
            all_features = concat_dicts(
                all_features,
                make_closest_feature(
                    i,
                    player,
                    "tomato",
                    self.get_tomato_dispenser_locations()
                    + counter_objects["tomato"],
                ),
            )
            all_features = concat_dicts(
                all_features,
                make_closest_feature(
                    i,
                    player,
                    "dish",
                    self.get_dish_dispenser_locations()
                    + counter_objects["dish"],
                ),
            )
            all_features = concat_dicts(
                all_features,
                make_closest_feature(
                    i, player, "soup", counter_objects["soup"]
                ),
            )
            all_features = concat_dicts(
                all_features,
                make_closest_feature(
                    i, player, "serving", self.get_serving_locations()
                ),
            )
            all_features = concat_dicts(
                all_features,
                make_closest_feature(
                    i,
                    player,
                    "empty_counter",
                    self.get_empty_counter_locations(overcooked_state),
                ),
            )

            # Closest pots info
            pot_locations = self.get_pot_locations().copy()
            for pot_idx in range(num_pots):
                _, closest_pot_loc = mlam.motion_planner.min_cost_to_feature(
                    player.pos_and_or, pot_locations, with_argmin=True
                )
                pot_features = make_pot_feature(
                    i, player, pot_idx, closest_pot_loc, pot_states
                )
                all_features = concat_dicts(all_features, pot_features)

                if closest_pot_loc:
                    pot_locations.remove(closest_pot_loc)

            # Adjacent features info
            for direction, pos_and_feat in enumerate(
                self.get_adjacent_features(player)
            ):
                _, feat = pos_and_feat
                all_features["p{}_wall_{}".format(i, direction)] = (
                    [0] if feat == " " else [1]
                )

        # Convert all list and tuple values to np.arrays
        features_np = {k: np.array(v) for k, v in all_features.items()}

        player_features = []  # Non-position player-specific features
        player_absolute_positions = []  # Position player-specific features
        player_relative_positions = (
            []
        )  # Relative position player-specific features

        # Compute all player-centric features for each player
        for i, player_i in enumerate(overcooked_state.players):
            # All absolute player-centric features
            player_i_dict = {
                k: v
                for k, v in features_np.items()
                if k[:2] == "p{}".format(i)
            }
            features = np.concatenate(list(player_i_dict.values()))
            abs_pos = np.array(player_i.position)

            # Calculate position relative to all other players
            rel_pos = []
            for player_j in overcooked_state.players:
                if player_i == player_j:
                    continue
                pj_rel_to_pi = np.array(
                    pos_distance(player_j.position, player_i.position)
                )
                rel_pos.append(pj_rel_to_pi)
            rel_pos = np.concatenate(rel_pos)

            player_features.append(features)
            player_absolute_positions.append(abs_pos)
            player_relative_positions.append(rel_pos)

        # Compute a symmetric, player-centric encoding of features for each player
        ordered_features = []
        for i, player_i in enumerate(overcooked_state.players):
            player_i_features = player_features[i]
            player_i_abs_pos = player_absolute_positions[i]
            player_i_rel_pos = player_relative_positions[i]
            other_player_features = np.concatenate(
                [feats for j, feats in enumerate(player_features) if j != i]
            )
            player_i_ordered_features = np.squeeze(
                np.concatenate(
                    [
                        player_i_features,
                        other_player_features,
                        player_i_rel_pos,
                        player_i_abs_pos,
                    ]
                )
            )
            ordered_features.append(player_i_ordered_features)

        return ordered_features

    return featurize_state


def create_featurize_2(self, mlam):
    def featurize_state_2(overcooked_state, num_pots=2, **kwargs):
        """
        Encode state with some manually designed features. Works for arbitrary number of players

        Arguments:
            overcooked_state (OvercookedState): state we wish to featurize
            mlam (MediumLevelActionManager): to be used for distance computations necessary for our higher-level feature encodings
            num_pots (int): Encode the state (ingredients, whether cooking or not, etc) of the 'num_pots' closest pots to each player.
                If i < num_pots pots are reachable by player i, then pots [i+1, num_pots] are encoded as all zeros. Changing this
                impacts the shape of the feature encoding

        Returns:
            ordered_features (list[np.Array]): The ith element contains a player-centric featurized view for the ith player

            The encoding for player i is as follows:

                [player_i_features, other_player_features player_i_dist_to_other_players, player_i_position]

                player_{i}_features (length num_pots*10 + 24):
                    pi_orientation: length 4 one-hot-encoding of direction currently facing
                    pi_obj: length 4 one-hot-encoding of object currently being held (all 0s if no object held)
                    pi_wall_{j}: {0, 1} boolean value of whether player i has wall immediately in direction j
                    pi_closest_{onion|tomato|dish|soup|serving|empty_counter}: (dx, dy) where dx = x dist to item, dy = y dist to item. (0, 0) if item is currently held
                    pi_cloest_soup_n_{onions|tomatoes}: int value for number of this ingredient in closest soup
                    pi_closest_pot_{j}_exists: {0, 1} depending on whether jth closest pot found. If 0, then all other pot features are 0. Note: can
                        be 0 even if there are more than j pots on layout, if the pot is not reachable by player i
                    pi_closest_pot_{j}_{is_empty|is_full|is_cooking|is_ready}: {0, 1} depending on boolean value for jth closest pot
                    pi_closest_pot_{j}_{num_onions|num_tomatoes}: int value for number of this ingredient in jth closest pot
                    pi_closest_pot_{j}_cook_time: int value for seconds remaining on soup. -1 if no soup is cooking
                    pi_closest_pot_{j}: (dx, dy) to jth closest pot from player i location

                other_player_features (length (num_players - 1)*(num_pots*10 + 24)):
                    ordered concatenation of player_{j}_features for j != i

                player_i_dist_to_other_players (length (num_players - 1)*2):
                    [player_j.pos - player_i.pos for j != i]

                player_i_position (length 2)
        """

        all_features = {}

        def concat_dicts(a, b):
            return {**a, **b}

        def make_closest_feature(idx, player, name, locations):
            """
            Compute (x, y) deltas to closest feature of type `name`, and save it in the features dict
            """
            feat_dict = {}
            obj = None
            held_obj = player.held_object
            held_obj_name = held_obj.name if held_obj else "none"
            if held_obj_name == name:
                obj = held_obj
                feat_dict["p{}_closest_{}".format(i, name)] = (0, 0)
            else:
                loc, deltas = self.get_deltas_to_closest_location(
                    player, locations, mlam
                )
                if loc and overcooked_state.has_object(loc):
                    obj = overcooked_state.get_object(loc)
                feat_dict["p{}_closest_{}".format(idx, name)] = deltas

            if name == "soup":
                num_onions = num_tomatoes = 0
                if obj:
                    ingredients_cnt = Counter(obj.ingredients)
                    num_onions, num_tomatoes = (
                        ingredients_cnt["onion"],
                        ingredients_cnt["tomato"],
                    )
                feat_dict["p{}_closest_soup_n_onions".format(i)] = [num_onions]
                feat_dict["p{}_closest_soup_n_tomatoes".format(i)] = [
                    num_tomatoes
                ]

            return feat_dict

        def make_pot_feature(idx, player, pot_idx, pot_loc, pot_states):
            """
            Encode pot at pot_loc relative to 'player'
            """
            # Pot doesn't exist
            feat_dict = {}
            if not pot_loc:
                feat_dict["p{}_closest_pot_{}_exists".format(idx, pot_idx)] = [
                    0
                ]
                feat_dict[
                    "p{}_closest_pot_{}_is_empty".format(idx, pot_idx)
                ] = [0]
                feat_dict[
                    "p{}_closest_pot_{}_is_full".format(idx, pot_idx)
                ] = [0]
                feat_dict[
                    "p{}_closest_pot_{}_is_cooking".format(idx, pot_idx)
                ] = [0]
                feat_dict[
                    "p{}_closest_pot_{}_is_ready".format(idx, pot_idx)
                ] = [0]
                feat_dict[
                    "p{}_closest_pot_{}_num_onions".format(idx, pot_idx)
                ] = [0]
                feat_dict[
                    "p{}_closest_pot_{}_num_tomatoes".format(idx, pot_idx)
                ] = [0]
                feat_dict[
                    "p{}_closest_pot_{}_cook_time".format(idx, pot_idx)
                ] = [0]
                feat_dict["p{}_closest_pot_{}".format(idx, pot_idx)] = (0, 0)
                return feat_dict

            # Get position information
            deltas = self.get_deltas_to_location(player, pot_loc)

            # Get pot state info
            is_empty = int(pot_loc in self.get_empty_pots(pot_states))
            is_full = int(pot_loc in self.get_full_pots(pot_states))
            is_cooking = int(pot_loc in self.get_cooking_pots(pot_states))
            is_ready = int(pot_loc in self.get_ready_pots(pot_states))

            # Get soup state info
            num_onions = num_tomatoes = 0
            cook_time_remaining = 0
            if not is_empty:
                soup = overcooked_state.get_object(pot_loc)
                ingredients_cnt = Counter(soup.ingredients)
                num_onions, num_tomatoes = (
                    ingredients_cnt["onion"],
                    ingredients_cnt["tomato"],
                )
                cook_time_remaining = (
                    0 if soup.is_idle else soup.cook_time_remaining
                )

            # Encode pot and soup info
            feat_dict["p{}_closest_pot_{}_exists".format(idx, pot_idx)] = [1]
            feat_dict["p{}_closest_pot_{}_is_empty".format(idx, pot_idx)] = [
                is_empty
            ]
            feat_dict["p{}_closest_pot_{}_is_full".format(idx, pot_idx)] = [
                is_full
            ]
            feat_dict["p{}_closest_pot_{}_is_cooking".format(idx, pot_idx)] = [
                is_cooking
            ]
            feat_dict["p{}_closest_pot_{}_is_ready".format(idx, pot_idx)] = [
                is_ready
            ]
            feat_dict["p{}_closest_pot_{}_num_onions".format(idx, pot_idx)] = [
                num_onions
            ]
            feat_dict[
                "p{}_closest_pot_{}_num_tomatoes".format(idx, pot_idx)
            ] = [num_tomatoes]
            feat_dict["p{}_closest_pot_{}_cook_time".format(idx, pot_idx)] = [
                cook_time_remaining
            ]
            feat_dict["p{}_closest_pot_{}".format(idx, pot_idx)] = deltas

            return feat_dict

        IDX_TO_OBJ = ["onion", "soup", "dish", "tomato"]
        OBJ_TO_IDX = {o_name: idx for idx, o_name in enumerate(IDX_TO_OBJ)}

        counter_objects = self.get_counter_objects_dict(overcooked_state)
        pot_states = self.get_pot_states(overcooked_state)

        for i, player in enumerate(overcooked_state.players):
            # Player info
            orientation_idx = Direction.DIRECTION_TO_INDEX[player.orientation]
            all_features["p{}_orientation".format(i)] = np.eye(4)[
                orientation_idx
            ]
            obj = player.held_object

            if obj is None:
                held_obj_name = "none"
                all_features["p{}_objs".format(i)] = np.zeros(len(IDX_TO_OBJ))
            else:
                held_obj_name = obj.name
                obj_idx = OBJ_TO_IDX[held_obj_name]
                all_features["p{}_objs".format(i)] = np.eye(len(IDX_TO_OBJ))[
                    obj_idx
                ]

            # Closest feature for each object type
            all_features = concat_dicts(
                all_features,
                make_closest_feature(
                    i,
                    player,
                    "onion",
                    self.get_onion_dispenser_locations()
                    + counter_objects["onion"],
                ),
            )
            all_features = concat_dicts(
                all_features,
                make_closest_feature(
                    i,
                    player,
                    "tomato",
                    self.get_tomato_dispenser_locations()
                    + counter_objects["tomato"],
                ),
            )
            all_features = concat_dicts(
                all_features,
                make_closest_feature(
                    i,
                    player,
                    "dish",
                    self.get_dish_dispenser_locations()
                    + counter_objects["dish"],
                ),
            )
            all_features = concat_dicts(
                all_features,
                make_closest_feature(
                    i, player, "soup", counter_objects["soup"]
                ),
            )
            all_features = concat_dicts(
                all_features,
                make_closest_feature(
                    i, player, "serving", self.get_serving_locations()
                ),
            )
            all_features = concat_dicts(
                all_features,
                make_closest_feature(
                    i,
                    player,
                    "empty_counter",
                    self.get_empty_counter_locations(overcooked_state),
                ),
            )

            # Closest pots info
            pot_locations = self.get_pot_locations().copy()
            for pot_idx in range(num_pots):
                _, closest_pot_loc = mlam.motion_planner.min_cost_to_feature(
                    player.pos_and_or, pot_locations, with_argmin=True
                )
                pot_features = make_pot_feature(
                    i, player, pot_idx, closest_pot_loc, pot_states
                )
                all_features = concat_dicts(all_features, pot_features)

                if closest_pot_loc:
                    pot_locations.remove(closest_pot_loc)

            # Adjacent features info
            for direction, pos_and_feat in enumerate(
                    self.get_adjacent_features(player)
            ):
                _, feat = pos_and_feat
                all_features["p{}_wall_{}".format(i, direction)] = (
                    [0] if feat == " " else [1]
                )

        # Convert all list and tuple values to np.arrays
        features_np = {k: np.array(v) for k, v in all_features.items()}

        player_features = []  # Non-position player-specific features
        player_absolute_positions = []  # Position player-specific features
        player_relative_positions = (
            []
        )  # Relative position player-specific features

        # Compute all player-centric features for each player
        for i, player_i in enumerate(overcooked_state.players):
            # All absolute player-centric features
            player_i_dict = {
                k: v
                for k, v in features_np.items()
                if k[:2] == "p{}".format(i)
            }
            features = np.concatenate(list(player_i_dict.values()))
            abs_pos = np.array(player_i.position)

            # Calculate position relative to all other players
            rel_pos = []
            for player_j in overcooked_state.players:
                if player_i == player_j:
                    continue
                pj_rel_to_pi = np.array(
                    pos_distance(player_j.position, player_i.position)
                )
                rel_pos.append(pj_rel_to_pi)
            rel_pos = np.concatenate(rel_pos)

            player_features.append(features)
            player_absolute_positions.append(abs_pos)
            player_relative_positions.append(rel_pos)

        # Compute a symmetric, player-centric encoding of features for each player
        ordered_features = []
        for i, player_i in enumerate(overcooked_state.players):
            player_i_features = player_features[i]
            player_i_abs_pos = player_absolute_positions[i]
            player_i_rel_pos = player_relative_positions[i]
            other_player_features = np.concatenate(
                [feats for j, feats in enumerate(player_features) if j != i]
            )
            player_i_ordered_features = np.squeeze(
                np.concatenate(
                    [
                        player_i_features,
                        other_player_features,
                        player_i_rel_pos,
                        player_i_abs_pos,
                    ]
                )
            )
            ordered_features.append(player_i_ordered_features)

        return ordered_features

    return featurize_state_2


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
                                              "PPO_asymmetric_advantages"),
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
