from tqdm import tqdm

from sotopia.database.logs import EpisodeLog

episode_pks = EpisodeLog.all_pks()
episode_pks_list: list[str] = list(episode_pks)
all_episodes = []
for pk in tqdm(episode_pks_list):
    try:
        curr_ep = EpisodeLog.get(pk)
    except:
        continue
    all_episodes.append(curr_ep)
models = ["togethercomputer/llama-2-70b-chat", "gpt-3.5-turbo"]
all_episodes_model_pairs = [
    ep
    for ep in all_episodes
    if (
        ep.models == ["gpt-4", models[1], models[0]]
        or ep.models == ["gpt-4", models[0], models[1]]
    )
]
for ep in all_episodes_model_pairs:
    ep.update(tag=f"{models[0]}_{models[1]}_v0.0.1")  # type: ignore

print("Done, the new tag is:", f"{models[0]}_{models[1]}_v0.0.1")
