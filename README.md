# RL-TCP

For years, TCP has been explored and discussed for end-to-end congestion control mechanisms. Although, various efforts and optimization techniques have taken place to find an optimal approach in dealing with all kinds of network architecture and scenarios. But the TCP version only follows the rules which they are defined by. The rule-based approach takes no consideration of the previous information of the links every time a flw begins,whereas a better performance can be accomplished if TCP can adapt its attitude based on prior learned information when the same path was before experienced. The rising of machine learning and deep learning paves the way for many researchers in academia. These approaches rather learn from history to find the best way and methods to get the job done.
This project discusses the implementation of Reinforcement Learning (RL) based TCP congestion control python Agent which uses RL based techniques to somehow overcome the rule-based shortcomings of TCP.

# RL Tools
The following RL tools are used for the implementation of the project.

# NS-3
Ns-3 is a discrete-event network simulator for networking systems, that primarily used for research and education. It became a standard in networking research in recent years as the results obtained are accepted by the science community. In this project, we used NS-3 to generate traffic and also simulate our environment. It can be found here. https://www.nsnam.org

# Ns-3 gym
ns3-gym is a middleware which interconnects ns-3 network simulator and OpenAI Gym framework.ns-3 gym works as a proxy between both ns-3 environments that is a C++ environment and the python Agent of Reinforcement learning which is in python. It can be installed from here. https://github.com/tkn-tub/ns3-gym

# Agent
Our agent interacts with the network environments of ns-3 and keeps exploring the optimal policy by taking various actions such as increasing or decreasing cWnd size. The environment setup is a combination of ns-3 network simulator, ns3-gym proxy, and a python based agent. Our mapping approach is as follows:
• State: The state space is the profile of our networking simulation that is provided by the ns-3 simulator.
• Action: The actions are considered to increase, decrease or leave no-change the cWnd size with +600, -200, 0 respectively.
• Reward: The reward is a utility function that considers the RTT and segments acknowledged parameters. +2 incase of increasing cWnd and not overwhelming the receiver side, otherwise -30.

As the result indicates, our agent adaptively adjusts the cWnd in a constant range of values according to an objective the agent realized.
