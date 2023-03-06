import pandas
import apyori

dataset = pandas.read_csv("../../data/Market_Basket_Optimisation.csv", header=None)

transactions = []
for i in range(0, 7501):
    product_list = []
    for j in range(0, 20):
        product_list.append(str(dataset.values[i, j]))
    transactions.append(product_list)

# we want products that appear in at least 3 transactions per day.
# over the full week 3*7 = 21 transactions must be found
# 7501 were taken over 1 week, so 21/7501 = 0.002799
#
# min length and max length 2 are because we're interested in "buy one, get one free" type deals
rules = apyori.apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift=3,
                       min_length=2, max_length=2)
results = list(rules)


# Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))


data_frame_results = pandas.DataFrame(inspect(results),
                                      columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

# Displaying the results non-sorted
print(data_frame_results)

# Displaying the results sorted by descending lifts
print(data_frame_results.nlargest(n=10, columns='Lift'))
