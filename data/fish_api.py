import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def getFishData():
    # csv로드------------------------------------------
    bream_length = pd.read_csv('bream_length.csv').to_numpy().flatten()
    # print(bream_length)
    bream_weight = pd.read_csv('bream_weight.csv').to_numpy().flatten()
    smelt_length = pd.read_csv('smelt_length.csv').to_numpy().flatten()
    smelt_weight = pd.read_csv('smelt_weight.csv').to_numpy().flatten()

    bream_data = np.column_stack((bream_length, bream_weight))
    smelt_data = np.column_stack((smelt_length, smelt_weight))
    # print(bream_data)
    # print(smelt_data)

    # 시각화---------------------------------------------
    # plt.scatter(bream_data[:,0],bream_data[:,1])
    # plt.scatter(smelt_data[:,0],smelt_data[:,1])
    # plt.xlabel("length")
    # plt.ylabel("weight")
    # plt.show()

    # 도미와 빙어 데이터 합치기-------------------------------------
    fish_data = np.concatenate((bream_data,smelt_data))
    # print(fish_data.shape)

    # 타겟 데이터 생성-------------------------------------------
    fish_target = np.concatenate((np.ones(35),np.zeros(14)))
    # print(fish_target)

    fish_target = fish_target.reshape((49,-1)) 
    fishs = np.hstack((fish_data, fish_target))
    # print(fishs)

    # 셔플해서 테스트 데이터와 훈련 데이터로 구분-------------------------
    np.random.seed(42)
    index = np.arange(49) # 35(도미), 14(빙어)
    np.random.shuffle(index)
    # print(index)

    # 35개
    train_input = fish_data[index[:35]] # 훈련 데이터 (모델)
    train_target = fish_target[index[:35]] # 타겟 데이터 (모델)
    # print(train_input)
    # print(train_target)
    train_fishs = np.hstack((train_input, train_target))
    # print(train_fishs)

    # 14개
    test_input = fish_data[index[35:]] # 훈련 데이터 (검증)
    test_target = fish_target[index[35:]] # 타겟 데이터 (검증)
    test_fishs = np.hstack((test_input, test_target))
    # print(test_fishs)
    new_fishs =  np.vstack((train_fishs, test_fishs)) # 셔플해서 훈련 검증으로 나뉜 데이터를 합쳐서 다시 만든 fishs
    # print(type(new_fishs)) 
    # 시각화----------------------------------------------------
    # plt.scatter(train_input[:,0],train_input[:,1])
    # plt.scatter(test_input[:,0],test_input[:,1])
    # plt.xlabel("length")
    # plt.ylabel("weight")
    # plt.show()

    return new_fishs

   


