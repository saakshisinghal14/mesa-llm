import pytest
from mesa.model import Model
from mesa.space import MultiGrid
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.reasoning.react import ReActReasoning

class DummyModel(Model):
    def __init__(self):
        super().__init__(rng=42)
        self.grid = MultiGrid(3, 3, torus=False)

def test_generate_obs_with_none_pos(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    model = DummyModel()
    
    agent = LLMAgent.create_agents(
        model,
        n=1,
        reasoning=ReActReasoning,
        system_prompt="Test prompt",
        vision=1,  # Must be > 0 to trigger _build_observation vision checks
        internal_state=[]
    ).to_list()[0]
    
    # Agent is explicitly NOT placed on the grid:
    agent.pos = None
    if hasattr(agent, "cell"):
        delattr(agent, "cell")

    # Mock memory to prevent API calls inside add_to_memory
    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *args, **kwargs: None)

    # Should not raise TypeError: 'NoneType' object is not iterable
    obs = agent.generate_obs()

    assert obs is not None
    assert obs.self_state["location"] is None
    assert len(obs.local_state) == 0
