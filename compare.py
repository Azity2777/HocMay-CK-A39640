import pandas as pd

list1 = pd.read_csv('decision_tree/decision_tree_predictions.csv', header=None).squeeze().tolist()
list2 = pd.read_csv('decision_tree/decision_tree_predictions_1.csv', header=None).squeeze().tolist()

count = 0
print("Các vị trí khác nhau:")
for i, (val1, val2) in enumerate(zip(list1, list2)):
    if val1 != val2:
        count += 1
        print(f'Vị trí: {i} - File 1: {val1} - File 2: {val2}')

print("Count:", count)