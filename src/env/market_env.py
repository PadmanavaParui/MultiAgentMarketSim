class MarketEnvironment:
    """
    Placeholder for PettingZoo/Gym environment wrapper around the CLOB.
    To be fully implemented in Week 1-4.
    """
    def __init__(self, agents, tick_size=0.01):
        self.agents = agents
        self.tick_size = tick_size
        
    def reset(self, seed=None):
        pass
        
    def step(self, actions):
        pass
        
    def observation_space(self, agent_id):
        pass
        
    def action_space(self, agent_id):
        pass
