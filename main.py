from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import random

class SugarAgent(Agent):
  def __init__(self, unique_id, model):
    super().__init__(unique_id, model)
    self.resources = random.randint(0, 10)
    self.wealth = self.resources

  def step(self):
    # 社会交互
    neighbors = self.model.grid.get_neighbors(self.pos, moore=True)
    if neighbors:
      other_agent = random.choice(neighbors)
      transfer_amount = random.randint(0, min(self.wealth, other_agent.wealth))
      self.wealth -= transfer_amount
      other_agent.wealth += transfer_amount

    # 阶层流动
    if random.random() < 0.1:
      self.wealth += 1

class SugarModel(Model):
  def __init__(self, width, height, N):
    self.num_agents = N
    self.grid = MultiGrid(width, height, True)
    self.schedule = RandomActivation(self)

    # 创建代理
    for i in range(self.num_agents):
      agent = SugarAgent(i, self)
      x = random.randrange(self.grid.width)
      y = random.randrange(self.grid.height)
      self.grid.place_agent(agent, (x, y))
      self.schedule.add(agent)

    # 数据收集
    self.datacollector = DataCollector(agent_reporters={"Wealth": "wealth"})

  def step(self):
    self.datacollector.collect(self)
    self.schedule.step()

# 模拟参数
width = 10
height = 10
num_agents = 100

# 创建模型
model = SugarModel(width, height, num_agents)

# 运行模拟
for i in range(100):
  model.step()

# 可视化数据
import matplotlib.pyplot as plt
agent_wealth = model.datacollector.get_agent_vars_dataframe().unstack(level=0)
agent_wealth.plot()
plt.show()
