from agent import Agent
from ddqn_agent import DDQN_Agent

if __name__ == "__main__":
    #agent = Agent(useGPU=True, useDepth=True)
    #agent.train()
    ddqn_agent = DDQN_Agent(useDepth=False)
    ddqn_agent.train()
