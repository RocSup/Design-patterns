import itertools
import random
import time

from sklearn import metrics
import pandas as pd
import numpy as np
import multiprocessing

from sklearn.utils import gen_even_slices


def cal(slice_distance, distances):
    splits_distances = metrics.pairwise_distances(slice_distance, distances)
    # splits_distances = pd.DataFrame()
    # for i in range(distances.shape[1]):
    #     distances.loc[:, i] += 1
    return splits_distances

def test1(tmp):
    a = tmp[0][1]
    return a

def test():
    tmp = np.zeros((50000, 50000))
    with multiprocessing.Pool(processes=4) as pool:
        res = pool.starmap(test1, zip(itertools.repeat(tmp)))
    print(res)

def start():
    arrs = []
    for i in range(40000):
        arrs.append([random.uniform(1.0, 15.0) for _ in range(50000)])
        # distances = pd.concat([distances, pd.DataFrame(pd.Series(splits), columns=i)], axis=1)
    distances = pd.DataFrame(arrs)
    # 计算每段的长度
    segment_length = distances.shape[0] // 4
    # 获取每段的开头和结尾数字
    segments_info = [(i * segment_length, (i + 1) * segment_length - 1) for i in range(4 - 1)]
    # 处理最后一段，确保总和为原始数字
    segments_info.append(((4 - 1) * segment_length, distances.shape[0] - 1))
    slice_distances = []
    for _, (start_number, end_number) in enumerate(segments_info):
        slice_distances.append(distances[start_number:end_number])
    start_time = time.time()
    with multiprocessing.Pool(processes=4) as pool:
        res = pool.starmap(cal, zip(slice_distances, itertools.repeat(distances)))

    # res = []
    # for split in slice_distances:
    #     splits_distances = metrics.pairwise_distances(split, distances)
    #     res.append(splits_distances)
    res1 = metrics.pairwise_distances(distances, distances)
    end_time = time.time()
    # res2 = metrics.pairwise_distances(distances, distances)
    # last_time = time.time()
    # res3 = metrics.pairwise_distances(distances[0:10000], distances)
    # last_time1 = time.time()
    print('time1:', end_time - start_time)
    # print('time2:', last_time - end_time)
    # print('time3:', last_time1 - last_time)
    # print(distances)
    # metrics.pairwise_distances()

    # metrics.pairwise.cosine_similarity

if __name__ == '__main__':
    test()

# time1: 35.28234243392944
# time2: 57.54812431335449
# time3: 15.49994421005249
