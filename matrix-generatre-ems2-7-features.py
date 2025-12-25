import os
import numpy as np
import pandas as pd
import argparse
import random
import multiprocessing
import torch
import esm

parser = argparse.ArgumentParser()
parser.add_argument('--seq_dir', required=True)
parser.add_argument('--type', required=True, metavar='predict or training or testing')
parser.add_argument('--dir', required=True, metavar='file')
parser.add_argument('--thread', required=True)
parser.add_argument('--subtype', required=True)
args = parser.parse_args()

def extract_esm(filename, subtype):
    EMB_PATH = '/path' + '/esm_seq'
    EMB_LAYER = 6
    fn = EMB_PATH + '/' + filename + '.pt'
    embs = torch.load(fn)
    Xs = (embs['representations'][EMB_LAYER])
    Xs = Xs.numpy()
    return Xs

class str_to_num():
    def __init__(self):
        self.seq1 = []
        self.seq2 = []
        self.name1 = []
        self.name2 = []
        self.label = []
        self.all_data = []

    def read_from_csv(self, csv_data, type='predict'):
        print("starting read data....")
        self.seq1 = csv_data['seq_1']
        self.seq2 = csv_data['seq_2']
        self.name1 = csv_data['new_name_1']
        self.name2 = csv_data['new_name_2']
        if type == 'predict':
            pass
        else:
            self.label = csv_data['label']
            assert (len(self.seq1) == len(self.label))
        assert (len(self.seq1) == len(self.seq2))
        print("read data done")

    def generate_seq_test(self, seq1, seq2, name1, name2, matchfile, subtype):
        info = pd.read_csv('/path' + '/' + subtype + '_HA1_forhi.csv')
        lista = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        listb = ['D', 'N']
        listz = ['E', 'Q']
        listj = ['L', 'I']

        strains = []
        one_char_1 = np.zeros((len(seq1), len(seq1[0]),7))
        one_char_2 = np.zeros((len(seq2), len(seq2[0]),7))
        for i in range(len(seq1)):
            for j in range(len(seq1[i])):
                seq1_aa = seq1[i][j]
                seq2_aa = seq2[i][j]
                if seq1[i][j] == 'X':
                    seq1_aa = random.choice(lista)
                elif seq1[i][j] == 'B':
                    seq1_aa = random.choice(listb)
                elif seq1[i][j] == 'Z':
                    seq1_aa = random.choice(listz)
                elif seq1[i][j] == 'J':
                    seq1_aa = random.choice(listj)
                if seq2[i][j] == 'X':
                    seq2_aa = random.choice(lista)
                elif seq2[i][j] == 'B':
                    seq2_aa = random.choice(listb)
                elif seq2[i][j] == 'Z':
                    seq2_aa = random.choice(listz)
                elif seq2[i][j] == 'J':
                    seq2_aa = random.choice(listj)

                one_char_1[i][j][0] = aj_dic[seq1_aa]['number']
                one_char_1[i][j][1] = aj_dic[seq1_aa]['access']
                one_char_1[i][j][2] = aj_dic[seq1_aa]['charge']
                one_char_1[i][j][3] = aj_dic[seq1_aa]['hydro']
                one_char_1[i][j][4] = aj_dic[seq1_aa]['hyindex']
                one_char_1[i][j][5] = aj_dic[seq1_aa]['polar']
                one_char_1[i][j][6] = aj_dic[seq1_aa]['volume']

                one_char_2[i][j][0] = aj_dic[seq2_aa]['number']
                one_char_2[i][j][1] = aj_dic[seq2_aa]['access']
                one_char_2[i][j][2] = aj_dic[seq2_aa]['charge']
                one_char_2[i][j][3] = aj_dic[seq2_aa]['hydro']
                one_char_2[i][j][4] = aj_dic[seq2_aa]['hyindex']
                one_char_2[i][j][5] = aj_dic[seq2_aa]['polar']
                one_char_2[i][j][6] = aj_dic[seq2_aa]['volume']

            ar1 = extract_esm(info['seqid'][info['Isolate_Id'] == name1[i]].iloc[0], subtype)
            ar2 = extract_esm(info['seqid'][info['Isolate_Id'] == name2[i]].iloc[0], subtype)
            strain = np.concatenate((ar1, one_char_1[i], ar2, one_char_2[i]), axis=1)
            strains.append(strain)
        mychar = np.array(strains).astype(np.float64)
        return mychar

    def generate_seq(self, seq1, seq2, name1, name2, aj_dic, subtype, label):
        print('begin')
        info = pd.read_csv('/path' + '/' + subtype + '_HA1_forhi.csv')
        lista = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        listb = ['D', 'N']
        listz = ['E', 'Q']
        listj = ['L', 'I']

        strains = []
        one_char_1 = np.zeros((len(seq1), len(seq1[0]),7))
        one_char_2 = np.zeros((len(seq2), len(seq2[0]),7))

        for i in range(len(seq1)):
            strain1 = []
            strain2 = []
            labels = []
            for j in range(len(seq1[i])):
                seq1_aa = seq1[i][j]
                seq2_aa = seq2[i][j]

                if seq1[i][j] == 'X':
                    seq1_aa = random.choice(lista)
                elif seq1[i][j] == 'B':
                    seq1_aa = random.choice(listb)
                elif seq1[i][j] == 'Z':
                     seq1_aa = random.choice(listz)
                elif seq1[i][j] == 'J':
                     seq1_aa = random.choice(listj)
                else:
                    pass
                if seq2[i][j] == 'X':
                   seq2_aa = random.choice(lista)
                elif seq2[i][j] == 'B':
                     seq2_aa = random.choice(listb)
                elif seq2[i][j] == 'Z':
                     seq2_aa = random.choice(listz)
                elif seq2[i][j] == 'J':
                     seq2_aa = random.choice(listj)
                else:
                    pass

                one_char_1[i][j][0] = aj_dic[seq1_aa]['number']
                one_char_1[i][j][1] = aj_dic[seq1_aa]['access']
                one_char_1[i][j][2] = aj_dic[seq1_aa]['charge']
                one_char_1[i][j][3] = aj_dic[seq1_aa]['hydro']
                one_char_1[i][j][4] = aj_dic[seq1_aa]['hyindex']
                one_char_1[i][j][5] = aj_dic[seq1_aa]['polar']
                one_char_1[i][j][6] = aj_dic[seq1_aa]['volume']

                one_char_2[i][j][0] = aj_dic[seq2_aa]['number']
                one_char_2[i][j][1] = aj_dic[seq2_aa]['access']
                one_char_2[i][j][2] = aj_dic[seq2_aa]['charge']
                one_char_2[i][j][3] = aj_dic[seq2_aa]['hydro']
                one_char_2[i][j][4] = aj_dic[seq2_aa]['hyindex']
                one_char_2[i][j][5] = aj_dic[seq2_aa]['polar']
                one_char_2[i][j][6] = aj_dic[seq2_aa]['volume']

                labels.append([label[i]])

            ar1 = extract_esm(info['seqid'][info['Isolate_Id'] == name1[i]].iloc[0], subtype)
            ar2 = extract_esm(info['seqid'][info['Isolate_Id'] == name2[i]].iloc[0], subtype)
            print(info['seqid'][info['Isolate_Id'] == name1[i]].iloc[0])
            print(ar1.shape)

            strain = np.concatenate((ar1, one_char_1[i], ar2, one_char_2[i], labels), axis=1)
            strains.append(strain)

        mychar = np.array(strains).astype(np.float64)
        return mychar

    def do_generate(self, type, matchfile, subtype):
        print('start generating seq....')
        if type == 'predict':
            self.all_data = self.generate_seq_test(self.seq1, self.seq2, self.name1, self.name2, matchfile, subtype)
        else:
            self.all_data = self.generate_seq(self.seq1, self.seq2, self.name1, self.name2, matchfile, subtype, self.label)
        print('generate seq done')

    def save_to_npy(self, dir, filename):
        arr = np.array(self.all_data)
        np.save(dir + '/' + filename + '.npy', arr)

def matrix_generate(record):
    seq_file, aj_dic, dir, type, filename, matchfile, subtype = record
    seq_all_0 = pd.read_csv(seq_file, sep='\t')
    seq_all = seq_all_0.reset_index(drop=True)
    s = str_to_num()
    s.read_from_csv(seq_all, type)
    s.do_generate(type, aj_dic, subtype)
    s.save_to_npy(dir, filename)
    print('done')

if __name__ == '__main__':
    import time
    start_time = time.time()

    aaindex=pd.read_csv("/path/aaindex_feature_BV.txt",sep='\t',index_col=0)
    aaindex1 = aaindex[['number']]
    matchfile = aaindex1.T
    dir = args.dir
    seq_dir = args.seq_dir
    all_dict = aaindex.T.to_dict()
    aj_dic = all_dict
    type = args.type
    subtype = args.subtype
    thread = int(args.thread)

    seq_file_list = os.listdir(seq_dir)
    print(seq_file_list)

    for seqfile in seq_file_list:
        filelist = [seqfile]
        pool = multiprocessing.Pool(processes=thread)
        for i in filelist:
            map_args = (seq_dir + '/' + i, aj_dic, dir, type, i,matchfile, subtype)
            pool.apply_async(matrix_generate, (map_args,))
        pool.close()
        pool.join()

    print("time is %s" % (time.time() - start_time))
