# -*- coding:utf-8 -*-
"""
Created on Thu 6 Apr 2017

author: Aiting Liu
"""
from __future__ import division
import os
import random

RAWDIR = './raw'
OUTPUT = './data'

if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)

file_list = ['0_simplifyweibo.txt', '1_simplifyweibo.txt', '2_simplifyweibo.txt', '3_simplifyweibo.txt']
output_file = 'all_simplifyweibo.txt'
data_set = ['train.txt', 'valid.txt', 'test.txt']


def process(file_path):
    fw = open(os.path.join(RAWDIR, output_file), 'w')

    for raw_file in file_path:
        f = open(os.path.join(RAWDIR, raw_file), 'r')
        lines = f.readlines()
        f.close()
        label_class = raw_file[0]
        print(label_class)
        if str(label_class) == '0':
            label = '喜悦'
        elif str(label_class) == '1':
            label = '愤怒'
        elif str(label_class) == '2':
            label = '厌恶'
        elif str(label_class) == '3':
            label = '低落'
        sents = []
        poses = []
        sent_length = []
        for i in range(0, len(lines)):
            line = lines[i].strip()
            tmp = line.split(' ')
            # print(tmp)
            sent = []
            pos = []
            for pair in tmp:
                tmp1 = pair.split('/')
                if len(tmp1) == 2:
                    # print(tmp1)
                    sent_tmp = tmp1[0]
                    pos_tmp = tmp1[1]
                    sent.append(sent_tmp)
                    pos.append(pos_tmp)
            sents.append(str(label) + ' ' + ' '.join(sent) + '\n')
            poses.append(str(label) + ' ' + ' '.join(pos) + '\n')
            sent_length.append(len(tmp))
        print(sent_length)
        print(max(sent_length))
        print(sum(sent_length)/len(sent_length))
        fw.writelines(sents)
    fw.close()


def split_data(all_data):
    """split all data into train_set/valid_set/test_set."""
    with open(os.path.join(RAWDIR, all_data), 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
        print(len(lines))
        train_number = len(lines) - 2000
        valid_number = 1000
        train_set = lines[:train_number]
        print(train_number, valid_number)
        valid_set = lines[train_number:train_number+valid_number]
        test_set = lines[train_number+valid_number:]
        fw = open(os.path.join(RAWDIR, 'train.txt'), 'w')
        fw.writelines(train_set)
        fw.close()
        fw = open(os.path.join(RAWDIR, 'valid.txt'), 'w')
        fw.writelines(valid_set)
        fw.close()
        fw = open(os.path.join(RAWDIR, 'test.txt'), 'w')
        fw.writelines(test_set)
        fw.close()


def get_data_set(data_set_path):
    OUTPUTDIR = ''

    if 'train' in data_set_path:
        OUTPUTDIR = os.path.join(OUTPUT, 'train')
        if not os.path.exists(OUTPUTDIR):
            os.makedirs(OUTPUTDIR)
    elif 'test' in data_set_path:
        OUTPUTDIR = os.path.join(OUTPUT, 'test')
        if not os.path.exists(OUTPUTDIR):
            os.makedirs(OUTPUTDIR)
    elif 'valid' in data_set_path:
        OUTPUTDIR = os.path.join(OUTPUT, 'valid')
        if not os.path.exists(OUTPUTDIR):
            os.makedirs(OUTPUTDIR)

    with open(os.path.join(RAWDIR, data_set_path), 'r') as f:
        lines = f.readlines()
        sents = []
        labels = []
        for line in lines:
            tmp = line.strip().split(' ')
            label = tmp[0]
            sent = tmp[1:]
            sents.append(' '.join(sent) + '\n')
            labels.append(label + '\n')
        fw = open(os.path.join(OUTPUTDIR, data_set_path.split('.')[0]+'_sent.txt'), 'w')
        fw.writelines(sents)
        fw.close()
        fw = open(os.path.join(OUTPUTDIR, data_set_path.split('.')[0] + '_label.txt'), 'w')
        fw.writelines(labels)
        fw.close()


def main():
    process(file_list)
    split_data(output_file)
    for d in data_set:
        get_data_set(d)
    print('Done.')

if __name__ == '__main__':
    main()

