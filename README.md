# Autonomous-Drone-Battery-Management-for-Urban-Surveillance-using-DQN-and-DDQN

## Scenario :
A single autonomous surveillance drone is operating over a simulated urban area. The drone's primary mission is to collect surveillance data by observing points of interest. It has a limited battery and needs to return to one of several charging stations when low on power. The challenge is to manage battery life effectively while maximizing surveillance opportunities, which appear randomly and offer varying rewards. Atmospheric disturbances (wind, rain) also unpredictably affect the drone's energy consumption.<br>
## Problem :
Statement: Develop a Deep Reinforcement Learning agent using Deep Q-Networks (DQN) and Double Deep Q-Networks (DDQN) to optimize the battery usage of an autonomous surveillance drone. The agent must learn an optimal policy to maximize its "surveillance score" by strategically flying, hovering, or returning to a charging station, while adapting to dynamic battery depletion, varying surveillance opportunities (rewards), and random atmospheric disturbances that affect energy consumption.<br>
## Key Concepts & State Representation:<br>
● Drone Position: The drone operates on a 2D grid representing the urban area (e.g., a 10×10 grid). Each cell in the grid represents a specific location.<br>
● Battery Level: A continuous value representing the percentage of battery remaining (e.g., 0% to 100%).<br>
● Charging Stations: Fixed locations on the grid where the drone can recharge.<br>
● Surveillance Opportunities (Points of Interest - POIs): These are temporary "hotspots" that appear randomly on the grid. Each POI has:<br>
&emsp;○ A value (reward) for observing it.<br>
&emsp;○ A lifespan (how many time steps it remains active before disappearing).<br>
● Atmospheric Disturbance: A continuous value representing the current level of wind/turbulence (e.g., from 0.0 to 1.0, where 1.0 means high disturbance). This directly affects energy consumption.<br>
## State Space for input to NN:<br>
● Drone X-coordinate: (Discrete, e.g., 0-9)<br>
● Drone Y-coordinate: (Discrete, e.g., 0-9)<br>
● Current Battery Percentage: (Continuous, e.g., float 0.0 to 100.0)<br>
● Current Atmospheric Disturbance: (Continuous, e.g., float 0.0 to 1.0)<br>
● Information about Active POIs: (For simplicity, you can limit to just the single "nearest" or "highest value" POI, or use a fixed number of top POIs)<br>
&emsp;○ Proximity to Nearest POI: (Continuous, e.g., Euclidean distance to closest active POI).<br>
&emsp;○ Value of Nearest POI: (Continuous, e.g., reward value).<br>
&emsp;○ Lifespan of Nearest POI: (Discrete, e.g., remaining timesteps).<br>
## Action Space:<br>
● Move North<br>
● Move South<br>
● Move East<br>
● Move West<br>
● Hover: Stay at the current location.<br>
● Recharge: (Only if at a Charging Station). Stay at the station to recharge.<br>
## Dynamics and Rewards:<br>
● Battery Depletion:<br>
&emsp;○ Moving: Costs Base_Move_Cost + (Atmospheric_Disturbance * Disturbance_Factor) per step.<br>
&emsp;○ Hovering: Costs Base_Hover_Cost + (Atmospheric_Disturbance * Disturbance_Factor) per step.<br>
&emsp;○ Recharging: Gain Recharge_Rate battery percentage per step.<br>
&emsp;○ Penalty: Large negative reward (-100) if battery reaches 0% and not at a charging station (drone crashes).<br>
● Surveillance Opportunities (POIs):<br>
&emsp;○ Spawning: New POIs appear randomly on empty cells (e.g., 5% chance per timestep).<br>
&emsp;○ Observation Reward: When the drone is at the exact location of an active POI, it collects the POI's value as a positive reward. The POI then disappears.<br>
&emsp;○ Lifespan Decay: Active POIs decay their lifespan by 1 each timestep; they disappear when lifespan reaches 0.<br>
● Atmospheric Disturbance: Changes randomly each time step (e.g., a small random walk, or jump to a new value with some probability). <br>
● Rewards:<br>
&emsp;○ Collecting POI value: +POI_Value.<br>
&emsp;○ Successful recharge (per step at station): Small positive reward (+1) to incentivize charging behavior.<br>
&emsp;○ Crash (battery 0% not at station): -100.<br>
&emsp;○ General time penalty: Small negative reward (-0.1) per step (to encourage efficient operation).<br>
## Objective:
The agent must learn the optimal policy π∗ using DQN and Double DQN to maximize its cumulative surveillance score (total rewards collected) over many episodes (e.g., each episode is 200-500 time steps).<br>
## Environment Setup (Custom Python Simulator):<br>
● A DroneSurveillanceEnv class with __init__, step, and reset methods.<br>
● The step method will handle:<br>
&emsp;○ Action execution (movement, battery change).<br>
&emsp;○ Battery depletion based on action and disturbance.<br>
&emsp;○ POI spawning, lifespan decay, and collection.<br>
&emsp;○ Atmospheric disturbance updates.<br>
&emsp;○ Reward calculation.<br>
&emsp;○ Episode termination conditions (battery crash, max timesteps).<br>
● The reset method will:<br>
&emsp;○ Place the drone at a starting location.<br>
&emsp;○ Reset battery to full.<br>
&emsp;○ Clear active POIs.<br>
&emsp;○ Set initial atmospheric disturbance.<br>

-------------------------------------------------------------------------------------------------------------------------------------------
1. Project Overview and Motivation

Urban surveillance using autonomous drones presents a practical challenge that goes beyond navigation alone. A drone must continuously balance its surveillance objectives with strict battery constraints while operating in a dynamic and uncertain environment. This project focuses on learning an intelligent battery management and navigation strategy for a surveillance drone operating over a city grid. The drone must decide where to move, when to hover, and when to recharge, while collecting valuable surveillance information from dynamically appearing points of interest (POIs). Reinforcement learning is used to allow the drone to learn these decisions through experience rather than relying on fixed, rule-based logic.

2. Problem Formulation as a Reinforcement Learning Task

The drone surveillance task is formulated as a sequential decision-making problem where the agent interacts with an environment over discrete time steps. At each step, the agent observes the drone’s current position, battery level, atmospheric conditions, and information about nearby POIs. Based on this state, it selects one action from a discrete set that includes movement, hovering, or recharging. The environment responds by updating the drone’s state and returning a reward that reflects the usefulness and safety of the chosen action. The objective is to maximize cumulative reward over an episode while avoiding battery depletion.

3. Environment Design and State Representation

The custom DroneSurveillanceEnv models a simplified urban grid where the drone operates. The environment tracks the drone’s position on a two-dimensional grid, its remaining battery, and the number of time steps elapsed in the episode. In addition, the environment includes dynamic POIs that appear randomly, each with a specific value and limited lifespan. Atmospheric disturbances are also modeled as a continuously changing variable that increases energy consumption. The state representation is compact but informative, consisting of seven elements that capture spatial position, normalized battery level, disturbance intensity, and key information about the nearest POI. This design provides the agent with enough context to make informed decisions without overwhelming the learning process.

4. Action Space and Drone Behavior

The drone is allowed to take six discrete actions: move north, south, east, west, hover in place, or recharge its battery. Movement actions change the drone’s position and consume battery, while hovering consumes a smaller amount of energy. Recharging is only possible at fixed charging stations located at the corners of the grid. This constraint forces the agent to plan ahead and manage its energy carefully, rather than reacting only when the battery is nearly depleted. The limited action space keeps the problem tractable while still allowing for meaningful behavior.

5. Battery Dynamics and Energy Modeling

Battery management is central to this project. Each movement or hover action consumes energy based on a base cost that is further influenced by atmospheric disturbance. Higher disturbance levels increase energy usage, making long-distance travel riskier. Recharging restores battery at a fixed rate but can only occur at charging stations. If the battery is depleted away from a charging station, the episode terminates with a severe penalty. This battery model encourages the agent to develop realistic behaviors such as early returns for charging and conservative movement during unfavorable conditions.

6. Reward Design and Learning Signal

The reward structure is designed to balance multiple objectives. The agent receives positive rewards for collecting POIs, reflecting successful surveillance. Small negative time penalties discourage unnecessary idling and promote efficiency. Recharging provides a small positive reward to reinforce good energy management habits. A large negative penalty is applied if the drone crashes due to battery depletion, strongly discouraging reckless behavior. Together, these rewards guide the agent toward strategies that prioritize valuable surveillance while maintaining safe battery levels.

7. Dynamic POIs and Environmental Uncertainty

Points of Interest are not static; they spawn randomly, have varying values, and expire after a limited lifespan. This introduces urgency and uncertainty into the task. The agent must decide whether pursuing a POI is worth the energy cost, especially when battery levels are low or disturbances are high. Atmospheric disturbances further increase unpredictability by changing energy costs over time. These elements prevent the environment from becoming deterministic and force the agent to learn adaptive strategies rather than memorizing fixed paths.

8. Deep Q-Network (DQN) Agent

The DQN agent approximates the action-value function using a neural network. Given a state, the network predicts Q-values for all possible actions, and the agent selects actions using an epsilon-greedy strategy. Experience replay is used to store transitions and sample them randomly during training, which helps break correlations between consecutive experiences. A separate target network is maintained to stabilize training by providing fixed targets during Q-value updates. This setup allows the agent to gradually learn effective navigation and battery management policies.

9. Double DQN for Improved Stability

While standard DQN is effective, it is known to suffer from Q-value overestimation. To address this, a Double DQN agent is also implemented. The key difference lies in how future rewards are estimated: the main network selects the best next action, while the target network evaluates that action’s value. This separation reduces overoptimistic value estimates and results in more stable learning. In practice, the Double DQN agent shows smoother learning curves and fewer sudden drops in performance compared to the standard DQN agent.

10. Training Process and Target Network Updates

Training proceeds over multiple episodes, each representing a full operational cycle of the drone. During each episode, the agent interacts with the environment, stores experiences in replay memory, and periodically updates the Q-network using mini-batches of past experiences. The target network is updated at fixed intervals to keep learning stable. Exploration gradually decreases over time as epsilon decays, allowing the agent to shift from exploration to exploitation once it has learned useful behaviors.

11. Performance Comparison between DQN and DDQN

Training results show clear differences between DQN and Double DQN. While both agents learn meaningful policies, Double DQN consistently demonstrates more stable performance with reduced variance in episode rewards. Standard DQN occasionally exhibits sharp drops in performance, which are likely caused by Q-value overestimation. The moving average plots further highlight that Double DQN converges more smoothly and reliably, making it a better choice for this environment.

12. Policy Behavior and Learned Strategies

Analysis of the learned policy reveals intuitive and adaptive behavior. When the battery level is high, the drone actively pursues high-value POIs, even if they are farther away. At medium battery levels, it becomes more selective, favoring nearby POIs that offer a good reward-to-energy ratio. When the battery is low, the agent prioritizes returning to the nearest charging station. The policy also adapts to atmospheric disturbances by avoiding long-distance travel during high disturbance periods and recharging earlier when energy costs increase.

13. Visualization and Behavioral Insights

The visualization function provides valuable insight into how the agent behaves during an episode. Trajectory plots show efficient movement patterns rather than random wandering. Battery level graphs clearly illustrate planned recharging behavior instead of last-moment recoveries. Disturbance plots demonstrate how environmental changes influence decisions, and cumulative reward plots confirm that the agent steadily accumulates value over time. These visualizations help validate that the learned policy aligns with the intended design goals.

14. Challenges Encountered

Several challenges were encountered during implementation. Designing a compact yet informative state representation required careful trade-offs, especially when deciding how much POI information to include. Reward shaping was also non-trivial, as small changes in penalty magnitudes significantly affected agent behavior. Training stability was another concern, particularly due to the non-stationary nature of POI spawning and atmospheric disturbances. These challenges reinforced the importance of careful environment design and algorithm choice.

15. Assumptions and Simplifications

To keep the problem manageable, several simplifying assumptions were made. The environment is modeled as a 2D grid with perfect positioning and no obstacles. Atmospheric disturbance is represented as a single scalar affecting all movements equally. The drone is assumed to have perfect sensing of POIs and instantaneous movement without inertia. Charging stations are always available and never congested. These assumptions allow the focus to remain on decision-making rather than low-level physical modeling.

16. Limitations of the Current Implementation

Despite its effectiveness, the implementation has limitations. Scaling the environment to larger grids or denser POI distributions would require more expressive state representations. The current single-drone setup does not account for coordination or competition between multiple agents. Additionally, the battery model is simplified and does not capture real-world effects such as temperature or battery aging. From an algorithmic perspective, discrete action DQN methods may struggle with finer-grained control required in real deployments.

17. Future Work and Extensions

There are several promising directions for future work. The environment could be extended to three dimensions with altitude control and more realistic wind models. Multi-drone scenarios could be explored to study coordination and resource sharing. More advanced reinforcement learning methods, such as hierarchical RL or continuous-control algorithms, could improve scalability and realism. Finally, transferring policies trained in simulation to real-world data would be an important step toward practical deployment.

18. Final Remarks

This project demonstrates how reinforcement learning can be applied to autonomous drone battery management in a dynamic surveillance setting. By combining realistic constraints, environmental uncertainty, and modern deep reinforcement learning algorithms, the agent learns intuitive and effective strategies for balancing mission objectives with energy limitations. While simplified, the system provides a strong foundation for more advanced and realistic autonomous drone applications.
