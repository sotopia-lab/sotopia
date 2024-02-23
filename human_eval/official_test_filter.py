import csv
import pandas as pd
import math
from rejson import Client, Path
import json
import redis
import scipy.stats

redis_host = 'tiger.lti.cs.cmu.edu'
redis_port = 6388
redis_password = 'aclkasjf29qwrUOIO'


selected_columns = [
    'player.pk',
    'player.data',
    'player.prolific_id',
    'player.believability_1',
    'player.believability_reasoning_1',
    'player.relationship_1',
    'player.relationship_reasoning_1',
    'player.knowledge_1',
    'player.knowledge_reasoning_1',
    'player.secret_1',
    'player.secret_reasoning_1',
    'player.social_rules_1',
    'player.social_rules_reasoning_1',
    'player.financial_and_material_benefits_1',
    'player.financial_and_material_benefits_reasoning_1',
    'player.goal_1',
    'player.goal_reasoning_1',
    'player.believability_2',
    'player.believability_reasoning_2',
    'player.relationship_2',
    'player.relationship_reasoning_2',
    'player.knowledge_2',
    'player.knowledge_reasoning_2',
    'player.secret_2',
    'player.secret_reasoning_2',
    'player.social_rules_2',
    'player.social_rules_reasoning_2',
    'player.financial_and_material_benefits_2',
    'player.financial_and_material_benefits_reasoning_2',
    'player.goal_2',
    'player.goal_reasoning_2'
]


def filter_out_useless_column(df):
    df = df[selected_columns]
    return df


def filter_out_useless_data(df):
    for col in selected_columns:
        if col in df.keys():
            df = df[df[col].notna()]
    return df


def filter_incomplete_pk(df):
    # filter out the pk has less than 2 complete data
    pk_list = df['player.pk'].tolist()
    pk_count = {}
    for pk in pk_list:
        if pk in pk_count:
            pk_count[pk] += 1
        else:
            pk_count[pk] = 1
    complete_pk = []
    for pk, count in pk_count.items():
        if count == 2:
            complete_pk.append(pk)
    df = df[df['player.pk'].isin(complete_pk)]
    return df



if __name__ == '__main__':
    source_file_name = './sotopia_official_study_2024-02-14_GPT4_representative_new_round.csv'
    target_file_name = './sotopia_official_study_2024-02-14_GPT4_representative_new_round_filtered.csv'
    df = pd.read_csv(source_file_name)
    df = filter_out_useless_data(df)
    df = filter_out_useless_column(df)
    df = filter_incomplete_pk(df)
    df.to_csv(target_file_name)