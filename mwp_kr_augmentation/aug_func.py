import re
import argparse
import numpy as np
from aug_func import *
import pandas as pd

def read_excel(filename, q_col_name):
    data = pd.read_excel(filename,  engine='openpyxl')
    pd_data = data.drop('Unnamed: 0', axis = 'columns') 
    q_org_ko = pd_data[q_col_name]
    return pd_data, q_org_ko

def replace_word(src_q):
    src_q = src_q.replace('㎝','cm')
    src_q = src_q.replace('센티미터','cm')
    src_q = src_q.replace('프리즘','각기둥')
    src_q = src_q.replace('큐브','각기둥')
    src_q = src_q.replace('입방체','정사각형')
    src_q = src_q.replace('입방센티미터','세제곱센티미터')
    src_q = src_q.replace('(앙스트롬)','(A)')
    src_q = src_q.replace('(ㄱ)','(A)')
    src_q = src_q.replace('(A)','(가)')
    src_q = src_q.replace('(a)','(가)')
    src_q = src_q.replace('(B)','(나)')
    src_q = src_q.replace('(b)','(나)')
    src_q = src_q.replace('(C)','(다)')
    src_q = src_q.replace('(c)','(다)')
    src_q = src_q.replace('(D)','(라)')
    src_q = src_q.replace('(d)','(라)')
    src_q = src_q.replace('(E)','(마)')
    src_q = src_q.replace('(e)','(마)')
    dst_q = src_q.replace('센티미터','cm')
    return dst_q


def find_en_noun(org_sentence, aug_sentence):
    org_en_nouns = list(filter(None, re.sub('[^a-zA-Z]', ' ', org_sentence).split(' ')))
    try:
        aug_en_nouns = list(filter(None, re.sub('[^a-zA-Z]', ' ', aug_sentence).split(' ')))
    except:
        aug_en_nouns = []
    return org_en_nouns, aug_en_nouns


def find_number(org_sentence, aug_sentence):
    org_nums = re.findall(r'\d+', org_sentence)
    try:
        aug_nums = re.findall(r'\d+', aug_sentence)
    except:
        aug_nums = []
    return org_nums, aug_nums


