import csv
import pandas as pd
import math


columns_to_filter = [
    'sotopia_pilot_study.1.player.prolific_id',
    'sotopia_pilot_study.1.player.believability_1',
    'sotopia_pilot_study.1.player.believability_1_gth',
    'sotopia_pilot_study.1.player.believability_reasoning_1',
    'sotopia_pilot_study.1.player.relationship_1',
    'sotopia_pilot_study.1.player.relationship_1_gth',
    'sotopia_pilot_study.1.player.relationship_reasoning_1',
    'sotopia_pilot_study.1.player.knowledge_1',
    'sotopia_pilot_study.1.player.knowledge_1_gth',
    'sotopia_pilot_study.1.player.knowledge_reasoning_1',
    'sotopia_pilot_study.1.player.secret_1',
    'sotopia_pilot_study.1.player.secret_1_gth',
    'sotopia_pilot_study.1.player.secret_reasoning_1',
    'sotopia_pilot_study.1.player.social_rules_1',
    'sotopia_pilot_study.1.player.social_rules_1_gth',
    'sotopia_pilot_study.1.player.social_rules_reasoning_1',
    'sotopia_pilot_study.1.player.financial_and_material_benefits_1',
    'sotopia_pilot_study.1.player.financial_and_material_benefits_1_gth',
    'sotopia_pilot_study.1.player.financial_and_material_benefits_reasoning_1',
    'sotopia_pilot_study.1.player.goal_1',
    'sotopia_pilot_study.1.player.goal_1_gth',
    'sotopia_pilot_study.1.player.goal_reasoning_1',
    'sotopia_pilot_study.1.player.believability_2',
    'sotopia_pilot_study.1.player.believability_2_gth',
    'sotopia_pilot_study.1.player.believability_reasoning_2',
    'sotopia_pilot_study.1.player.relationship_2',
    'sotopia_pilot_study.1.player.relationship_2_gth',
    'sotopia_pilot_study.1.player.relationship_reasoning_2',
    'sotopia_pilot_study.1.player.knowledge_2',
    'sotopia_pilot_study.1.player.knowledge_2_gth',
    'sotopia_pilot_study.1.player.knowledge_reasoning_2',
    'sotopia_pilot_study.1.player.secret_2',
    'sotopia_pilot_study.1.player.secret_2_gth',
    'sotopia_pilot_study.1.player.secret_reasoning_2',
    'sotopia_pilot_study.1.player.social_rules_2',
    'sotopia_pilot_study.1.player.social_rules_2_gth',
    'sotopia_pilot_study.1.player.social_rules_reasoning_2',
    'sotopia_pilot_study.1.player.financial_and_material_benefits_2',
    'sotopia_pilot_study.1.player.financial_and_material_benefits_2_gth',
    'sotopia_pilot_study.1.player.financial_and_material_benefits_reasoning_2',
    'sotopia_pilot_study.1.player.goal_2',
    'sotopia_pilot_study.1.player.goal_2_gth',
    'sotopia_pilot_study.1.player.goal_reasoning_2'
]


def filter_out_useless_data(df):
    for col in columns_to_filter:
        if col in df.keys():
            df = df[df[col].notna()]
    return df

def choose_qualified_ones(df):
    import json, ast
    pilot_study_data = pd.read_csv('./pilot_study_data.csv')
    pilot_study_list = []
    for pk, processed_data in zip(pilot_study_data['pk'], pilot_study_data['processed_data']):
        data_dict = ast.literal_eval(processed_data)
        pilot_study_list.append((pk, data_dict))

    for idx, row in df.iterrows():
        player_data = row['sotopia_pilot_study.1.player.data']
        actual_dict = ast.literal_eval(player_data)
        for pilot_data in pilot_study_list:
            if pilot_data[-1]['scenario'] == actual_dict['scenario'] and pilot_data[-1]['names'] == tuple(actual_dict['names']):
                player_data_pk = pilot_data[0]
                df.at[idx, 'pk'] = player_data_pk
                break
    
    pilot_study_reference = pd.read_csv('./pilot_study_reference.csv')
    pilot_study_reference = pilot_study_reference.to_dict(orient='records')
    player_data = df.to_dict(orient='records')

    comparing_columns = [
        'sotopia_pilot_study.1.player.believability_1',
        'sotopia_pilot_study.1.player.believability_1_gth',
        'sotopia_pilot_study.1.player.believability_reasoning_1',
        'sotopia_pilot_study.1.player.relationship_1',
        'sotopia_pilot_study.1.player.relationship_1_gth',
        'sotopia_pilot_study.1.player.relationship_reasoning_1',
        'sotopia_pilot_study.1.player.knowledge_1',
        'sotopia_pilot_study.1.player.knowledge_1_gth',
        'sotopia_pilot_study.1.player.knowledge_reasoning_1',
        'sotopia_pilot_study.1.player.secret_1',
        'sotopia_pilot_study.1.player.secret_1_gth',
        'sotopia_pilot_study.1.player.secret_reasoning_1',
        'sotopia_pilot_study.1.player.social_rules_1',
        'sotopia_pilot_study.1.player.social_rules_1_gth',
        'sotopia_pilot_study.1.player.social_rules_reasoning_1',
        'sotopia_pilot_study.1.player.financial_and_material_benefits_1',
        'sotopia_pilot_study.1.player.financial_and_material_benefits_1_gth',
        'sotopia_pilot_study.1.player.financial_and_material_benefits_reasoning_1',
        'sotopia_pilot_study.1.player.goal_1',
        'sotopia_pilot_study.1.player.goal_1_gth',
        'sotopia_pilot_study.1.player.goal_reasoning_1',
        'sotopia_pilot_study.1.player.believability_2',
        'sotopia_pilot_study.1.player.believability_2_gth',
        'sotopia_pilot_study.1.player.believability_reasoning_2',
        'sotopia_pilot_study.1.player.relationship_2',
        'sotopia_pilot_study.1.player.relationship_2_gth',
        'sotopia_pilot_study.1.player.relationship_reasoning_2',
        'sotopia_pilot_study.1.player.knowledge_2',
        'sotopia_pilot_study.1.player.knowledge_2_gth',
        'sotopia_pilot_study.1.player.knowledge_reasoning_2',
        'sotopia_pilot_study.1.player.secret_2',
        'sotopia_pilot_study.1.player.secret_2_gth',
        'sotopia_pilot_study.1.player.secret_reasoning_2',
        'sotopia_pilot_study.1.player.social_rules_2',
        'sotopia_pilot_study.1.player.social_rules_2_gth',
        'sotopia_pilot_study.1.player.social_rules_reasoning_2',
        'sotopia_pilot_study.1.player.financial_and_material_benefits_2',
        'sotopia_pilot_study.1.player.financial_and_material_benefits_2_gth',
        'sotopia_pilot_study.1.player.financial_and_material_benefits_reasoning_2',
        'sotopia_pilot_study.1.player.goal_2',
        'sotopia_pilot_study.1.player.goal_2_gth',
        'sotopia_pilot_study.1.player.goal_reasoning_2'
    ]

    qualified_annotators = []
    for data in player_data:
        qualified = True
        prolific_id = data['sotopia_pilot_study.1.player.prolific_id']
        for ref in pilot_study_reference:
            if data['pk'] == ref['PK']:
                for column in comparing_columns:
                    if 'reasoning' not in column and '_gth' not in column:
                        ref_column_name = column.split('.')[-1]
                        mask = df['pk'] == ref['PK']
                        df.loc[mask, column + '_gth'] = ref[ref_column_name]
                        if abs(data[column] - ref[ref_column_name]) > 2:
                            qualified = False
        if qualified is True:
            qualified_annotators.append(prolific_id)
    return qualified_annotators, df


df = pd.read_csv('./all_apps_wide_2024-01-28_2.csv')
df = filter_out_useless_data(df)
qualified_annotators, df = choose_qualified_ones(df)
import pdb; pdb.set_trace()

df = df[columns_to_filter]

df.to_csv('./filtered_prolific-01-28_2.csv')

print(qualified_annotators)