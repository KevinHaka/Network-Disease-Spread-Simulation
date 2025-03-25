from joblib import Parallel, delayed
from filelock import FileLock
from ruamel.yaml import YAML
from textwrap import dedent
from io import BytesIO

import matplotlib.pyplot as plt
import imageio.v2 as imageio
import networkx as nx
import numpy as np
import random

class Network:
    """
    Represents a network of nodes and edges used to simulate the spread of a disease.

    Attributes:
        config (dict): Configuration parameters for the network.
        has_spread (bool): Indicates whether the disease has spread beyond a threshold.
        nodes_per_state (dict): Number of nodes in each state.
        infections (int): Total number of infected nodes.
        size (int): Total number of nodes in the network.
        graph (networkx.Graph): The underlying graph structure of the network.
    """

    def __init__(self, config):
        """
        Initializes the Network object with the given configuration.

        Args:
            config (dict): Configuration parameters for the network.
        """

        self.config = config
        self.has_spread = False
        self.nodes_per_state = {state: self.config['initial_nodes_per_state'][state] 
                                for state in self.config['states']}
        self.infections = sum(self.nodes_per_state[infection_state]
                              for infection_state in self.config['infected_states'])
        self.size = sum(self.nodes_per_state.values())
        self.graph = self.initialize_graph()
        self.config["visualization"]["adjusted_node_scaling_factor"] = \
        self.config["visualization"]["node_scaling_factor"] / max(dict(self.graph.degree()).values())

    def initialize_graph(self):
        """
        Initializes the graph by creating it, assigning states, and computing positions.

        Returns:
            networkx.Graph: The initialized graph.
        """

        graph = self.create_graph()
        states = self.generate_node_states()
        positions = self.compute_positions(graph)

        self.assign_node_attributes(graph, states, positions)
        return graph

    def create_graph(self):
        """
        Creates a graph based on the specified type and parameters in the configuration.

        Returns:
            networkx.Graph: The created graph.

        Raises:
            ValueError: If the graph type is unsupported.
        """

        graph_type = self.config["graph_type"]
        graph_parameters = self.config["graph_parameters"]

        if (graph_type == 'ER') or (graph_type == 'R'):
            return nx.erdos_renyi_graph(self.size, graph_parameters['p'])
        elif (graph_type == 'WS') or (graph_type == 'SW'):
            return nx.watts_strogatz_graph(self.size, graph_parameters['k'], graph_parameters['p'])
        elif (graph_type == 'BA') or (graph_type == 'SF'):
            return nx.barabasi_albert_graph(self.size, graph_parameters['m'])
        else:
            message = dedent(f"""
                Unsupported graph type: {graph_type}
                Supported types are
                - `ER` or `R` for Erdos-Renyi graph (Random graph)
                - `WS` or `SW` for Watts-Strogatz graph (Small-World graph)
                - `BA` or `SF` for Barabasi-Albert graph (Scale-Free graph)
            """)
            raise ValueError(message)

    def compute_positions(self, graph):
        """
        Computes the positions of nodes in the graph using a spring layout.

        Args:
            graph (networkx.Graph): The graph for which positions are computed.

        Returns:
            dict: A dictionary of node positions.
        """

        return nx.spring_layout(graph)

    def generate_node_states(self):
        """
        Generates the initial states for the nodes based on the configuration.

        Returns:
            list: A list of states for the nodes.
        """

        initial_nodes_per_state = self.config["initial_nodes_per_state"]
        state_placement_strategy = self.config["state_placement_strategy"]

        states_without_strategy = []
        state_with_strategy = state_placement_strategy[0] if state_placement_strategy else ""

        for state, count in initial_nodes_per_state.items():
            if state != state_with_strategy:
                states_without_strategy.extend([state]*count)
        return states_without_strategy

    def assign_node_attributes(self, graph, states, positions):
        """
        Assigns attributes (state and position) to the nodes in the graph.

        Args:
            graph (networkx.Graph): The graph to which attributes are assigned.
            states (list): The states to assign to the nodes.
            positions (dict): The positions of the nodes.
        """

        state_placement_strategy = self.config["state_placement_strategy"]

        shuffled_nodes = list(graph.nodes())
        random.shuffle(shuffled_nodes)
        
        if state_placement_strategy:
            state_with_strategy, strategy = state_placement_strategy
            num_strategy_nodes = self.config["initial_nodes_per_state"][state_with_strategy]

            match strategy:
                case "D":
                    degree_centrality = nx.degree_centrality(graph)
                    sorted_nodes  = sorted(shuffled_nodes, key=lambda node: 
                                           degree_centrality[node], reverse=True)
                case "C":
                    closeness_centrality = nx.closeness_centrality(graph)
                    sorted_nodes  = sorted(shuffled_nodes, key=lambda node: 
                                           closeness_centrality[node], reverse=True)
                case "B":
                    betweenness_centrality = nx.betweenness_centrality(graph)
                    sorted_nodes  = sorted(shuffled_nodes, key=lambda node: 
                                           betweenness_centrality[node], reverse=True)
                case _:
                    message = dedent(f"""
                        Unsupported strategy: {strategy}
                        Supported strategies are
                        - `D` for Degree centrality
                        - `C` for Closeness centrality
                        - `B` for Betweenness centrality
                    """)
                    raise ValueError(message)
            
            strategy_nodes = sorted_nodes [:num_strategy_nodes]
            
            for node in strategy_nodes: 
                graph.nodes[node]['position'] = positions[node]
                graph.nodes[node]['state'] = state_with_strategy
                shuffled_nodes.remove(node)

        for node, state in zip(shuffled_nodes, states):
            graph.nodes[node]['position'] = positions[node]
            graph.nodes[node]['state'] = state
    
    def draw_graph(self):
        """
        Draws the graph with nodes and edges styled based on their attributes.

        Returns:
            matplotlib.figure.Figure: The figure object containing the graph visualization.
        """

        bg = self.config["visualization"]["background_color"]
        adj_node_scaling_factor = self.config["visualization"]["adjusted_node_scaling_factor"]
        figsize = self.config["visualization"]["figure_size"]
        color_per_state = self.config["visualization"]["color_per_state"]
        
        graph = self.graph
        num_edges = len(graph.edges()) 
        edge_width = 1 / (np.log10(num_edges + 1) + 1)

        node_positions = [position for _, position in graph.nodes(data='position')]
        node_colors = [color_per_state[state] for _, state in graph.nodes(data='state')]
        node_sizes = [adj_node_scaling_factor * (degree + 1) for _, degree in graph.degree()]

        min_x, min_y = np.min(node_positions, axis=0)
        max_x, max_y = np.max(node_positions, axis=0)
        x_pad = 0.05*(max_x - min_x)
        y_pad = 0.05*(max_y - min_y)

        figure = plt.figure(figsize=figsize)
        plt.xlim((min_x - x_pad, max_x + x_pad))
        plt.ylim((min_y - y_pad, max_y + y_pad))

        nx.draw(graph, pos=node_positions, node_color=node_colors, 
                node_size=node_sizes, width=edge_width)
        figure.set_facecolor(bg)

        return figure

    def diffusion(self):
        """
        Simulates the diffusion process in the network.

        This method calculates the edges and weights, determines the total number of contacts,
        samples weighted edges, processes the contacts, handles state transitions, and checks
        if the disease has spread beyond a threshold.
        """

        edges, weights = self.calculate_edges_and_weights()
        total_contacts  = self.calculate_contacts()
        contacts = self.sample_weighted_edges(total_contacts, edges, weights)

        self.process_contacts(contacts)
        self.process_state_transitions()
        self.check_spread()

    def calculate_contacts(self):
        """
        Calculates the total number of contacts in the network based on the configuration.

        Returns:
            int: The total number of contacts.
        """

        contacts_per_state = self.config["contacts_per_state"]

        exact_total_contacts = sum([contacts * self.nodes_per_state[state] 
                                    for state, contacts in contacts_per_state.items()])
        exact_total_contacts /= 2

        total_contacts = int(exact_total_contacts)
        if random.random() < (exact_total_contacts - total_contacts): total_contacts += 1

        return total_contacts 

    def process_contacts(self, contacts):
        """
        Processes the contacts between nodes to simulate disease transmission.

        Args:
            contacts (list): A list of edges representing contacts between nodes.
        """

        initial_infected_state = self.config["initial_infected_state"]
        infection_pairs = self.config["infection_pairs"]
        infection_matrix = self.config["infection_matrix"]

        for edge in contacts:
            node1, node2 = edge
            node1_state = self.graph.nodes[node1]['state']
            node2_state = self.graph.nodes[node2]['state']
            pair = (node1_state, node2_state)

            if pair in infection_pairs:
                infection = infection_matrix[node1_state][node2_state]
                random_number = random.random()

                if random_number < abs(infection):
                    node = node1 if infection > 0 else node2

                    self.update_node_state(node, initial_infected_state)
                    self.infections += 1
    
    def calculate_edges_and_weights(self):
        """
        Calculates the edges and their corresponding weights based on node states.

        Returns:
            tuple: A tuple containing a list of edges and a list of weights.
        """

        precomputed_weights = self.config["precomputed_weights"]
        edges = list(self.graph.edges())

        weights = [precomputed_weights[(self.graph.nodes[node1]['state'], self.graph.nodes[node2]['state'])]
                    for node1, node2 in edges]
        
        return edges, weights

    def process_state_transitions(self):
        """
        Processes state transitions for nodes based on the transition matrix.

        Nodes in transitory states may transition to other states based on probabilities.
        Dead states result in the removal of edges connected to the node.
        """

        transitory_states = self.config["transitory_states"]
        transition_matrix = self.config["transition_matrix"]
        dead_states = self.config["dead_states"]

        for node in self.graph.nodes():
            node_state = self.graph.nodes[node]['state']

            if node_state not in transitory_states: continue

            cumulative_probability = 0
            random_number = random.random()
            
            for target_state, probability in transition_matrix[node_state].items():
                cumulative_probability += probability

                if random_number < cumulative_probability:
                    self.update_node_state(node, target_state)

                    if target_state in dead_states: 
                        self.delete_edges(node)
                    break

    def update_node_state(self, node, new_state):
        """
        Updates the state of a node and adjusts the count of nodes per state.

        Args:
            node (int): The node to update.
            new_state (str): The new state to assign to the node.
        """

        node_state = self.graph.nodes[node]['state']

        self.graph.nodes[node]['state'] = new_state
        self.nodes_per_state[node_state] -= 1
        self.nodes_per_state[new_state] += 1

    def check_spread(self):
        """
        Checks if the disease has spread beyond a predefined threshold.

        Updates the `has_spread` attribute if the threshold is exceeded.
        """

        if not self.has_spread:
            spreading_threshold = self.config["spreading_threshold"]
            self.has_spread = self.infections/self.size >= spreading_threshold
    
    def delete_edges(self, node):
        """
        Deletes all edges connected to a given node.

        Args:
            node (int): The node whose edges will be removed.
        """

        edges_to_remove = list(self.graph.edges(node))
        self.graph.remove_edges_from(edges_to_remove)
    
    @staticmethod
    def sample_weighted_edges(total_contacts, edges, weights):
        """
        Samples a set of edges based on their weights.

        Args:
            total_contacts (int): The total number of contacts to sample.
            edges (list): A list of edges in the graph.
            weights (list): A list of weights corresponding to the edges.

        Returns:
            list: A list of sampled edges.
        """

        weights_sum = np.sum(weights)

        if weights_sum > 0:
            cdf = np.cumsum(weights) / weights_sum
            random_values = np.random.random(total_contacts)
            indices = np.searchsorted(cdf, random_values, side='right')
            sampled_edges = [edges[idx] for idx in indices]
            
        else:
            sampled_edges = []
    
        return sampled_edges
    
    @staticmethod
    def configure_parameters(raw_config, simulation_time_step):
        """
        Configures and preprocesses the parameters for the network simulation.

        Args:
            raw_config (dict): The raw configuration dictionary for the network.
            simulation_time_step (float): The time step for the simulation.

        Returns:
            dict: The updated configuration dictionary with preprocessed parameters.
        """

        raw_config = raw_config.copy()

        daily_contacts_per_state = raw_config['daily_contacts_per_state']
        raw_transition_matrix = raw_config['raw_transition_matrix']
        infection_matrix = raw_config['infection_matrix']

        # Compute contacts and weights
        contacts_per_state, precomputed_weights = Network.compute_contacts_and_weights(
            daily_contacts_per_state, simulation_time_step)

        # Build the transition matrix
        transition_matrix = Network.build_transition_matrix(
            raw_transition_matrix, simulation_time_step)

        # Setup infection rules
        new_infection_matrix, infection_pairs = Network.setup_infection_rules(
            infection_matrix)

        # Identify special states
        transitory_states, dead_states = Network.identify_special_states(
            contacts_per_state, transition_matrix)
        
        states = list(contacts_per_state.keys())
        
        # Update the configuration dictionary
        raw_config['contacts_per_state'] = contacts_per_state
        raw_config['precomputed_weights'] = precomputed_weights
        raw_config['transition_matrix'] = transition_matrix
        raw_config['infection_matrix'] = new_infection_matrix
        raw_config['infection_pairs'] = infection_pairs
        raw_config['transitory_states'] = transitory_states
        raw_config['dead_states'] = dead_states
        raw_config['states'] = states

        return raw_config

    @staticmethod
    def compute_contacts_and_weights(daily_contacts_per_state, simulation_time_step):
        """
        Computes the number of contacts per state and precomputed weights for edges.

        Args:
            daily_contacts_per_state (dict): Daily contacts for each state.
            simulation_time_step (float): The time step for the simulation.

        Returns:
            tuple: A tuple containing:
                - contacts_per_state (dict): The number of contacts per state.
                - precomputed_weights (dict): Precomputed weights for edges.
        """

        contacts_per_state = {}
        precomputed_weights = {}

        for state1, contacts1 in daily_contacts_per_state.items():
            contacts_per_state[state1] = simulation_time_step * contacts1

            for state2, contacts2 in daily_contacts_per_state.items():
                precomputed_weights[(state1, state2)] = contacts1 * contacts2

        return contacts_per_state, precomputed_weights
    
    @staticmethod
    def build_transition_matrix(raw_transition, simulation_time_step):
        """
        Builds the transition matrix for state transitions.

        Args:
            raw_transition (dict): The raw transition matrix.
            simulation_time_step (float): The time step for the simulation.

        Returns:
            dict: The processed transition matrix.
        """

        states = list(raw_transition.keys())
        transition_matrix = {}
        
        for state_index, (current_state, transitions) in enumerate(raw_transition.items()):
            mean_days = transitions[state_index]

            leaving_probability = simulation_time_step / mean_days
            staying_probability = 1 - leaving_probability

            transition_matrix[current_state] = {
                target_state: (transitions[target_index] * leaving_probability 
                               if target_index != state_index else staying_probability) 
                               for target_index, target_state in enumerate(states)
            }

        return transition_matrix

    @staticmethod
    def setup_infection_rules(infection_matrix):
        """
        Sets up the infection rules based on the infection matrix.

        Args:
            infection_matrix (dict): The infection matrix defining infection probabilities.

        Returns:
            tuple: A tuple containing:
                - new_infection_matrix (dict): The processed infection matrix.
                - infection_pairs (list): A list of infection pairs.
        """

        states = list(infection_matrix.keys())
        new_infection_matrix = {}
        infection_pairs = []

        for state, infection_list in infection_matrix.items():
            new_infection_matrix[state] = {
                target_state: value for target_state, value in zip(states, infection_list)}

        for idx, state1 in enumerate(states):
            for state2 in states[idx:]:
                if state1 != state2:
                    new_infection_matrix[state2][state1] = -new_infection_matrix[state1][state2]

                    if new_infection_matrix[state1][state2] != 0:
                        infection_pairs.append((state1, state2))
                        infection_pairs.append((state2, state1))

                else:
                    new_infection_matrix[state1][state2] = 0
        
        return new_infection_matrix, infection_pairs
    
    @staticmethod
    def identify_special_states(contacts_per_state, transition_matrix):
        """
        Identifies transitory and dead states in the network.

        Args:
            contacts_per_state (dict): The number of contacts per state.
            transition_matrix (dict): The transition matrix for state transitions.

        Returns:
            tuple: A tuple containing:
                - transitory_states (list): A list of transitory states.
                - dead_states (list): A list of dead states.
        """

        transitory_states = []
        dead_states = []
        
        for state, contacts in contacts_per_state.items():
            transition_probability = transition_matrix[state][state]

            if transition_probability < 1:
                transitory_states.append(state)
            elif not contacts:
                dead_states.append(state)
    
        return transitory_states, dead_states

class Simulation:
    """
    Represents a simulation of disease spread on a network.

    Attributes:
        config (dict): Configuration parameters for the simulation.
        network_config (dict): Configuration parameters for the network.
        network (Network): The network object used in the simulation.
    """

    def __init__(self, config):
        """
        Initializes the Simulation object with the given configuration.

        Args:
            config (dict): Configuration parameters for the simulation and network.
        """

        self.config = config['simulation']
        self.network_config = Network.configure_parameters(
            config['network'], self.config['time_step'])
        self.network = None

    def run(self):
        """
        Runs the simulation using parallel processing.
        """

        monte_carlo_number = self.config['monte_carlo_number']
        jobs_number = self.config['jobs_number']
        seed = self.config['seed']

        np.random.seed(seed)
        seed_sequence = np.random.SeedSequence(seed)
        seeds = seed_sequence.spawn(monte_carlo_number)
        
        self._initialize_results_file()
        Parallel(jobs_number)(delayed(self._simulate)(seed.generate_state(1)[0], n)
                              for n, seed in enumerate(seeds, 1))                              

    def _simulate(self, seed, simulation_number):
        """
        Simulates the disease spread for a single Monte Carlo iteration.

        Args:
            seed (int): The random seed for this simulation.
            simulation_number (int): The index of the current simulation.
        """

        np.random.seed(seed)
        random.seed(int(seed))
        
        time_step = self.config['time_step']
        total_time = self.config['total_time']
        states_and_time = self.network_config['states']+['time']
        produce_gif = self.network_config['visualization']['produce_gif']
        is_last_simulation = simulation_number == self.config['monte_carlo_number']

        steps = round(total_time / time_step)
        data = {state: np.empty(steps+1) for state in states_and_time}

        # Initialize the network and collect initial results
        self.network = Network(self.network_config)
        self._collect_results(0, 0, data)

        if produce_gif and is_last_simulation:
            gif_filename = self.network_config['visualization']['gif_filename']
            gif_duration = self.network_config['visualization']['gif_duration']
            fps = self.network_config['visualization']['gif_fps']

            frames = [self._gif_frame(0)]
            steps_per_frame = steps / (gif_duration * fps)
            next_frame_step = steps_per_frame
        
        # Perform the simulation steps
        for step in range(1, steps+1):
            current_time = step * time_step

            self.network.diffusion()
            self._collect_results(step, current_time, data)
            
            if produce_gif and is_last_simulation:
                while step >= next_frame_step:
                    frames.append(self._gif_frame(current_time))
                    next_frame_step += steps_per_frame

        # Save the GIF if required
        if produce_gif and is_last_simulation :
            imageio.mimsave(gif_filename, frames, fps=fps, loop=0)

        # Write the results to the output file
        self._write_results(simulation_number, data)
    
    def _gif_frame(self, time):
        """
        Generates a single frame for the simulation GIF.

        Args:
            time (float): The current simulation time.

        Returns:
            numpy.ndarray: The image data for the frame.
        """

        figure = self.network.draw_graph()
        axes = figure.gca()
        axes.text(0.05, 0.95, f"Simulation time: {time:.1f} days", transform=axes.transAxes,
                  fontsize=12, verticalalignment='top', horizontalalignment='left', 
                  bbox=dict(facecolor='white', alpha=0.4))
        buffer = BytesIO()
        figure.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
        buffer.seek(0)
        frame = imageio.imread(buffer)

        buffer.close()
        plt.close(figure)

        return frame

    def _collect_results(self, step, time, data):
        """
        Collects the results of the simulation at a specific time step.

        Args:
            step (int): The current simulation step.
            time (float): The current simulation time.
            data (dict): The dictionary to store the results.
        """

        data['time'][step] = time

        for state, value in self.network.nodes_per_state.items():
            data[state][step] = value
    
    def _initialize_results_file(self):
        """
        Initializes the results file by writing the header row.

        The header includes the spread status, simulation number, time, and states.
        """

        data_filename = self.config['data_filename']
        states = self.network_config['states']
        txt_states = ",".join(states)

        with open(data_filename, "w") as f:
            f.write(f"spread_status,simulation,time,{txt_states}\n")

    def _write_results(self, simulation_number, data):
        """
        Writes the simulation results to the output file.

        Args:
            simulation_number (int): The index of the current simulation.
            data (dict): The dictionary containing the simulation results.
        """

        data_filename = self.config['data_filename']
        states_and_time = ['time']+self.network_config['states']
        
        spread_status = self.network.has_spread
        lock = FileLock(f"{data_filename}.lock")
        steps = len(data['time'])

        with lock:
            with open(data_filename, "a") as f:
                for step in range(steps):
                    f.write(f"{spread_status},{simulation_number}")

                    for state in states_and_time:
                        f.write(f",{data[state][step]}")
                    f.write("\n")

if __name__ == "__main__":
    yaml = YAML()
    
    # Load the configuration file
    with open('config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.load(file)

    # Run the simulation
    simulation = Simulation(config)
    simulation.run()