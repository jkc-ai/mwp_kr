import re
import argparse
import numpy as np
from aug_func import *

import pandas as pd
from googletrans import Translator

from collections import defaultdict
from textattack.augmentation import WordNetAugmenter, EmbeddingAugmenter, EasyDataAugmenter, CharSwapAugmenter, CheckListAugmenter, CLAREAugmenter

translator = Translator()

def parse_args():
    parser = argparse.ArgumentParser(description='input')
    parser.add_argument('--input-file', default='sample.xlsx', help='the dir to save logs and models')
    parser.add_argument('--aug-num',  type=int, default=3, help='the dir to save logs and models')

    args = parser.parse_args()
    return args

def refine_augmented_data(src_pd_data, aug_num):
    
    dst_pd_data = src_pd_data.copy()

    for num in range(aug_num):
        augmenter = 'clareaug_' + str(num)
        refine_q = []
        for n, or_q, aug_q in zip(src_pd_data['번호'], src_pd_data['문제'], src_pd_data[augmenter]):
            or_q = replace_word(or_q)
            aug_q = replace_word(aug_q)

            org_en_nouns, aug_en_nouns = find_en_noun(or_q, aug_q)
            intersection = list(set(aug_en_nouns).difference(set(org_en_nouns)))
            
            
            org_nums, aug_nums = find_number(or_q, aug_q)
            
            if org_nums == aug_nums and len(intersection) == 0:
                refine_q.append(aug_q)
            else:
                refine_q.append(np.nan)

        dst_pd_data[augmenter] =  refine_q  

    return dst_pd_data

def nlp_based_aug(src_pd_data, aug_num):
    dst_pd_data = src_pd_data.copy()

    clare_aug = CLAREAugmenter(pct_words_to_swap=0.1, transformations_per_example=aug_num)
    en_clareaug_kos = defaultdict(list)


    for n, or_q in zip(src_pd_data['번호'], src_pd_data['문제']):
        en_q = translator.translate(or_q, dest = 'en').text
        print(n)
        try:         
            en_clareaugs = clare_aug.augment(en_q)
        except:
            en_clareaugs = [None] * aug_num

        for num in range(aug_num):
            en_clareaug = en_clareaugs[num]
            en_clareaugaug_ko = translator.translate(en_clareaug, dest = 'ko').text
            en_clareaug_kos[num].append(en_clareaugaug_ko)
            print(en_clareaugaug_ko)

    for num in range(aug_num):
        dst_pd_data['clareaug_' + str(num)] = en_clareaug_kos[num]

    return dst_pd_data


def main():
    args = parse_args()

    input_file = args.input_file
    aug_num = args.aug_num

    ################################################################################
    q_col_name = '문제'
    pd_data, q_org_ko = read_excel(input_file, q_col_name)

    pd_data = nlp_based_aug(pd_data, aug_num)
    
    # ################################################################################
    refined_pd_data = refine_augmented_data(pd_data, aug_num)


    with pd.ExcelWriter(input_file) as writer:
        refined_pd_data.to_excel(writer)
        print("save %s" %input_file)

if __name__ == '__main__':
    main()
