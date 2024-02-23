import json
from collections import defaultdict
import random

with open('complete_gpt_score.json', 'r') as f:
    complete_gpt_score = json.load(f)

with open('complete_pk_agent_pairs.json', 'r') as f:
    complete_pk_agent_pairs = json.load(f)

# build env_dict for each environment collect all the data
env_dict = defaultdict(list)
for key, value in complete_gpt_score.items():
    env_dict[value['env_pk']].append({'agent1': value['agent1'], 'agent2': value['agent2'], 'pk': key, 'env_pk': value['env_pk']})

goal_score = []
for i in range(1000):
    # random sample 2 datapoint for each environment, need to random
    sampled_data = []
    for key, value in env_dict.items():
        scores = []
        for data in value:
            env_pk = data['env_pk']
            pk = data['pk']
            agent1_score = data['agent1']
            agent2_score = data['agent2']
            for pair in complete_pk_agent_pairs:
                if pair[0] == pk:
                    if pair[-1] == 'agent1':
                        agent1_score['pk'] = pk
                        scores.append(agent1_score)
                    else:
                        agent2_score['pk'] = pk
                        scores.append(agent2_score)
        sampled_data.extend(random.sample(scores, 2))

    average_score = {'believability': 0, 'relationship': 0, 'knowledge': 0, 'secret': 0, 'social_rules': 0, 'financial_and_material_benefits': 0, 'goal': 0, 'overall_score': 0}
    for dimension in average_score.keys():
        average_score[dimension] = sum([score[dimension] for score in sampled_data]) / len(sampled_data)
    goal_score.append(average_score['goal'])

    pks = []
    for data in sampled_data:
        print(data['pk'])
        pks.append(data['pk'])

    if abs(average_score['goal'] - 5.9) < 0.1:
        print(len(set(pks)))
        with open('pk_list.json', 'w') as f:
            json.dump(pks, f, indent=4)
        import pdb; pdb.set_trace()

print(sum(goal_score) / len(goal_score))
print(max(goal_score))

