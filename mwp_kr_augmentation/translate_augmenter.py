import re
import argparse
import numpy as np
from aug_func import *
import pandas as pd
from googletrans import Translator


translator = Translator()

def parse_args():

    parser = argparse.ArgumentParser(description='input')
    parser.add_argument('--input-file', default='sample.xlsx', help='the dir to save logs and models')
    parser.add_argument('--lang', default='en', help='the dir to save logs and models', nargs='+')
    # parser.add_argument('--list_data', type=int, nargs='+')

    args = parser.parse_args()
    return args


def retranslator_ko(lang, q_org_ko):
    trans_langs = []
    rekos = []
    for idx, i in enumerate(q_org_ko):        
        trans_lang = translator.translate(i, dest = lang).text
        reko = translator.translate(trans_lang, dest = 'ko').text
        trans_langs.append(trans_lang)
        rekos.append(reko)
    return trans_langs, rekos

def refine_augmented_data(lang, src_pd_data):
    refine_q = []
    dst_pd_data = src_pd_data.copy()
    for n, or_q, aug_q in zip(src_pd_data['번호'], src_pd_data['문제'], src_pd_data[lang + '-ko-aug']):
        or_q = replace_word(or_q)
        aug_q = replace_word(aug_q)

        org_en_nouns, aug_en_nouns = find_en_noun(or_q, aug_q)
        intersection = list(set(aug_en_nouns).difference(set(org_en_nouns)))
        
        
        org_nums, aug_nums = find_number(or_q, aug_q)
        
        if org_nums == aug_nums and len(intersection) == 0:
            refine_q.append(aug_q)
        else:
            refine_q.append(np.nan)

    dst_pd_data[lang + '-ko-aug'] =  refine_q  

    return dst_pd_data

def main():
    args = parse_args()

    input_file = args.input_file
    langs = args.lang
    print("langs: ", langs)
    ################################################################################
    q_col_name = '문제'
    pd_data, q_org_ko = read_excel(input_file, q_col_name)

    for lang in langs:

        trans_langs, rekos = retranslator_ko(lang, q_org_ko)

        # pd_data[lang] = trans_langs
        pd_data[lang + '-ko-aug'] = rekos

        ################################################################################
        refined_pd_data = refine_augmented_data(lang, pd_data)


    with pd.ExcelWriter(input_file) as writer:
        refined_pd_data.to_excel(writer)
        print("save %s" %input_file)

if __name__ == '__main__':
    main()
