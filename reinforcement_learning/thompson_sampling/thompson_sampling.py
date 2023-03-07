import matplotlib.pyplot as plt
import pandas
from math import log
from math import sqrt
from matplotlib import pyplot
import random

# click-through rate of ads
# In reality users connect one by one in real time, to simulate it the Ads CTR dataset will act as
# a test dummy as if we showed ads to users one by one
dataset = pandas.read_csv('../../data/Ads_CTR_Optimisation.csv')


# Helper utility to track ad selection, sums and perform UCB calculations
class ItemStat:
    def __init__(self):
        self.reward_0_count = 0
        self.reward_1_count = 0

    def increment_reward_0(self):
        self.reward_0_count += 1

    def increment_reward_1(self):
        self.reward_1_count += 1

    def random_draw(self):
        return random.betavariate(self.reward_1_count + 1, self.reward_0_count + 1)


num_users = 500
num_ads = 10
selected_ads = []

stats = [ItemStat() for _ in range(10)]
total_reward = 0

random.seed(0)

for round_index in range(0, num_users):
    print(f'---------round {round_index}----------')
    # search through all ad indexes and see which has the best estimated value
    selected_index = 0
    best_val = 0
    for advert_index in range(0, num_ads):
        current_val = stats[advert_index].random_draw()
        print(f'bound at {advert_index} is {current_val}')
        if current_val > best_val:
            print(f'current best is now {advert_index}')
            best_val = current_val
            selected_index = advert_index

    print(f'selected {selected_index}')
    selected_ads.append(selected_index)

    outcome = dataset.values[round_index, selected_index]
    if outcome == 0:
        stats[selected_index].increment_reward_0()
    else:
        stats[selected_index].increment_reward_1()

    total_reward = total_reward + outcome

pyplot.hist(selected_ads)
pyplot.title('Histogram of Ad Selections')
pyplot.xlabel('Ads')
pyplot.ylabel('Ad Selection Count')
plt.show()

print(f'total reward {total_reward}')
