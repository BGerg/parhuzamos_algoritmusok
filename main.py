import concurrent.futures
import numpy as np
import pandas as pd
import random
from collections import Counter
import matplotlib.pyplot as plt
from itertools import repeat
import multiprocessing as mp
import time

class KNN:

    def __init__(self):
        self.accuracy = 0.0
        self.predictions = 0
        self.accurate_of_predictions = 0


    def predict(self, training_data, to_predict):
        k = 3
        if len(training_data) >= k:
            print("K cannot be smaller than the total voting groups(ie. number of training data points)")
            return
        distributions = []
        for group in training_data:
            for features in training_data[group]:
                # Find euclidean distance using the numpy function
                euclidean_distance = np.linalg.norm(np.array(features) - np.array(to_predict))
                distributions.append([euclidean_distance, group])
        # Find the k nearest neighbors

        results = [i[1] for i in sorted(distributions)[:k]]
        # Figure out which is the most common class amongst the neighbors.
        result = Counter(results).most_common(1)[0][0]
        confidence = Counter(results).most_common(1)[0][1] / k

        return result, to_predict

    def test_multiprocess(self, test_set, training_set):
        pool = mp.Pool(processes=8)
        arr = {}

        s = time.process_time()
        for group in test_set:
            arr[group] = pool.starmap(self.predict, zip(repeat(training_set), test_set[group]))
        e = time.process_time()


        for group in test_set:
            for data in test_set[group]:
                for i in arr[group]:
                    if data == i[1]:
                        self.predictions += 1
                        if group == i[0]:
                            self.accurate_of_predictions += 1

        self.accuracy = 100 * (self.accurate_of_predictions / self.predictions)
        acurracy = str(self.accuracy)
        exec_time = e - s
        return exec_time, acurracy


    def test_multithread(self, test_set, training_set):

        arr_two = {}
        ez = []
        s = time.process_time()
        for group in test_set:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for thread in range(8):
                    futures.append(executor.submit(self.predict, training_set, test_set[group]))
                for future in concurrent.futures.as_completed(futures):
                    ez.append(future.result())

        #build same dict content format
            temp_list = []
            for i in range(len(ez[0][1])):
                values = ez[0][1][0]
                temp_touple = (group, values)
                temp_list.append((temp_touple))
            arr_two[group] = temp_list
            ez = []
        e = time.process_time()

        for group in test_set:
            for data in test_set[group]:
                for i in arr_two[group]:
                    if data == i[1]:
                        self.predictions += 1
                        if group == i[0]:
                            self.accurate_of_predictions += 1

        self.accuracy = 100 * (self.accurate_of_predictions / self.predictions)
        acurracy = str(self.accuracy)
        exec_time = e - s
        return exec_time, acurracy


def prepare_data(df):
    df.replace('?', -999999, inplace=True)
    df.replace('yes', 4, inplace=True)
    df.replace('no', 2, inplace=True)
    df.replace('notpresent', 4, inplace=True)
    df.replace('present', 2, inplace=True)
    df.replace('abnormal', 4, inplace=True)
    df.replace('normal', 2, inplace=True)
    df.replace('poor', 4, inplace=True)
    df.replace('good', 2, inplace=True)
    df.replace('ckd', 4, inplace=True)
    df.replace('notckd', 2, inplace=True)

def work_with_sort_csv():
    df = pd.read_csv(r"chronic_kidney_disease.csv")
    prepare_data(df)
    dataset = df.astype(float).values.tolist()
    random.shuffle(dataset)

    test_size = 0.2

    training_set = {2: [], 4: []}
    test_set = {2: [], 4: []}

    training_data = dataset[:-int(test_size * len(dataset))]
    test_data = dataset[-int(test_size * len(dataset)):]

    for record in training_data:
        training_set[record[-1]].append(
            record[:-1])
    for record in test_data:
        test_set[record[-1]].append(
            record[:-1])

    knn = KNN()
    multiprocess_time, multiprocess_acurracy = knn.test_multiprocess(test_set, training_set)
    multithread_time, multithread_acurracy= knn.test_multithread(test_set, training_set)

    print("\nAcurracy :", str(multiprocess_acurracy) + "%")
    print("Exec Time: ", multiprocess_time)
    print("\nAcurracy :", str(multithread_acurracy) + "%")
    print("Exec Time: ", multithread_time)
    return [multiprocess_time, multithread_time]

def work_with_long_csv():
    df = pd.read_csv(r"chronic_kidney_disease_long.csv")
    prepare_data(df)
    dataset = df.astype(float).values.tolist()
    random.shuffle(dataset)
    test_size = 0.2

    training_set = {2: [], 4: []}
    test_set = {2: [], 4: []}

    training_data = dataset[:-int(test_size * len(dataset))]
    test_data = dataset[-int(test_size * len(dataset)):]

    for record in training_data:
        training_set[record[-1]].append(
            record[:-1])
    for record in test_data:
        test_set[record[-1]].append(
            record[:-1])

    knn = KNN()
    multiprocess_time, multiprocess_acurracy = knn.test_multiprocess(test_set, training_set)
    multithread_time, multithread_acurracy= knn.test_multithread(test_set, training_set)

    print("\nAcurracy :", str(multiprocess_acurracy) + "%")
    print("Exec Time: ", multiprocess_time)
    print("\nAcurracy :", str(multithread_acurracy) + "%")
    print("Exec Time: ", multithread_time)
    return [multiprocess_time, multithread_time]

def main():
    result_short = work_with_sort_csv()
    result_long = work_with_long_csv()


    labels = ['400', '800']
    multiprocess = [result_short[0], result_short[1]]
    multithread = [result_long[0], result_long[1]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, multiprocess, width, label='multiprocess')
    rects2 = ax.bar(x + width / 2, multithread, width, label='multithread')


    ax.set_ylabel('Time')
    ax.set_xlabel('Input size')
    ax.set_title('Input size and time correlation')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()
if __name__ == "__main__":
    main()
