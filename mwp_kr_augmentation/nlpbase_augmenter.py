import re


import argparse
import numpy as np
from aug_func import *

import pandas as pd
from googletrans import Translator


from textattack.augmentation import WordNetAugmenter, EmbeddingAugmenter, EasyDataAugmenter, CharSwapAugmenter, CheckListAugmenter, CLAREAugmenter

translator = Translator()

def parse_args():
    parser = argparse.ArgumentParser(description='input')
    parser.add_argument('--input-file', default='sample.xlsx', help='the dir to save logs and models')

    args = parser.parse_args()
    return args

def refine_augmented_data(src_pd_data):
    
    dst_pd_data = src_pd_data.copy()

    for augmenter in ['wordnet', 'emb', 'emb', 'easy', 'charswap', 'checklist']:
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

def nlp_based_aug(src_pd_data):
    dst_pd_data = src_pd_data.copy()

    wordnet_aug = WordNetAugmenter()
    # en_wordnetaugs = []
    en_wordnetaug_kos = []

    emb_aug = EmbeddingAugmenter()
    # en_ebaugs = []
    en_embaug_kos = []

    easy_aug = EasyDataAugmenter(pct_words_to_swap=0.04)
    # en_easyaugs = []
    en_easyaug_kos = []

    charswap_aug = CharSwapAugmenter()
    # en_charswapaugs = []
    en_charswapaug_kos = []

    checklist_aug = CheckListAugmenter()
    # en_checklistaugs = []
    en_checklistaug_kos = []


    for n, or_q in zip(src_pd_data['번호'], src_pd_data['문제']):
         print(n)
         en_q = translator.translate(or_q, dest = 'en').text
         en_wordnetaug = wordnet_aug.augment(en_q)[0]
         wordnetaug_ko = translator.translate(en_wordnetaug, dest = 'ko').text
         en_wordnetaug_kos.append(wordnetaug_ko)
         print(wordnetaug_ko)

         emb_woaug = emb_aug.augment(en_q)[0]
         embaug_ko = translator.translate(emb_woaug, dest = 'ko').text
         en_embaug_kos.append(embaug_ko)
         print(embaug_ko)
         easy_woaug = easy_aug.augment(en_q)[0]
         easyaug_ko = translator.translate(easy_woaug, dest = 'ko').text
         en_easyaug_kos.append(easyaug_ko)
         print(easyaug_ko)
         charswap_woaug = charswap_aug.augment(en_q)[0]
         charswapaug_ko = translator.translate(charswap_woaug, dest = 'ko').text
         en_charswapaug_kos.append(charswapaug_ko)
         print(charswapaug_ko)
         checklist_woaug = checklist_aug.augment(en_q)[0]
         checklist_ko = translator.translate(checklist_woaug, dest = 'ko').text
         en_checklistaug_kos.append(checklist_ko)
         print(checklist_ko)


    dst_pd_data['wordnet'] = en_wordnetaug_kos
    dst_pd_data['emb'] = en_embaug_kos
    dst_pd_data['easy'] = en_easyaug_kos
    dst_pd_data['charswap'] = en_charswapaug_kos
    dst_pd_data['checklist'] = en_checklistaug_kos
    return dst_pd_data


def main():
    args = parse_args()

    input_file = args.input_file

    ################################################################################
    q_col_name = '문제'
    pd_data, q_org_ko = read_excel(input_file, q_col_name)

    pd_data = nlp_based_aug(pd_data)
    
    # ################################################################################
    refined_pd_data = refine_augmented_data(pd_data)


    with pd.ExcelWriter(input_file) as writer:
        refined_pd_data.to_excel(writer)
        print("save %s" %input_file)

    

if __name__ == '__main__':
    main()
