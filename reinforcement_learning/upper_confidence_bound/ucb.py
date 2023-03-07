import matplotlib.pyplot as plt
import pandas
from math import log
from math import sqrt
from matplotlib import pyplot

# click-through rate of ads
# In reality users connect one by one in real time, to simulate it the Ads CTR dataset will act as
# a test dummy as if we showed ads to users one by one
dataset = pandas.read_csv('../../data/Ads_CTR_Optimisation.csv')


# Helper utility to track ad selection, sums and perform UCB calculations
class ItemStat:
    def __init__(self, num_selected=0, reward_sum=0):
        self.num_selected = num_selected
        self.reward_sum = reward_sum

    def calc_average_reward(self):
        return self.reward_sum / self.num_selected

    def calc_deltaI(self, round_number):
        return sqrt((3 / 2) * log(round_number + 1) / self.num_selected)

    def calc_max_conf(self, round_number):
        if self.num_selected == 0:
            # using a large value to ensure every ad is selected at least once during the bounds calculation
            return 1e400

        return self.calc_average_reward() + self.calc_deltaI(round_number)

    def select(self):
        self.num_selected += 1

    def increment_rewards(self, reward):
        self.reward_sum = self.reward_sum + reward


num_users = 10000
num_ads = 10
stats = [ItemStat(), ItemStat(), ItemStat(), ItemStat(), ItemStat(), ItemStat(),
         ItemStat(), ItemStat(), ItemStat(), ItemStat()]
selected_ads = []
total_reward = 0

for round_index in range(0, num_users):
    print(f'---------round {round_index}----------')
    # search through all ad indexes and see which has the current highest confidence
    selected_index = 0
    max_confidence_bound = 0
    for advert_index in range(0, num_ads):
        current_bound = stats[advert_index].calc_max_conf(round_index)
        print(f'bound at {advert_index} is {current_bound}')
        if current_bound > max_confidence_bound:
            print(f'current best is now {advert_index}')
            max_confidence_bound = current_bound
            selected_index = advert_index

    print(f'selected {selected_index}')
    stats[selected_index].select()
    selected_ads.append(selected_index)

    outcome = dataset.values[round_index, selected_index]

    stats[selected_index].increment_rewards(outcome)
    total_reward = total_reward + outcome

pyplot.hist(selected_ads)
pyplot.title('Histogram of Ad Selections')
pyplot.xlabel('Ads')
pyplot.ylabel('Ad Selection Count')
plt.show()

print(f'total reward {total_reward}')
