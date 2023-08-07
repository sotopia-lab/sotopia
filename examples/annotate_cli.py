from typing import Generator, cast

import rich
from rich.layout import Layout
from rich.prompt import Prompt
from tqdm import tqdm

from sotopia.database import AnnotationForEpisode, Annotator, EpisodeLog

# get all the episode logs
episode_log_pks: Generator[str, None, None] = EpisodeLog.all_pks()

annotator_id = Prompt.ask("Enter your annotator id")
try:
    annotator = Annotator.get(annotator_id)
except:
    rich.print("[red] Annotator not found. Please create an annotator first.")
    name = Prompt.ask("Enter your name to sign up for annotation")
    email = Prompt.ask("Enter your email as well")
    Annotator(pk=annotator_id, name=name, email=email).save()
    rich.print("[green] Annotator created. Please run the script again.")
    exit()

annotated_episode = set()
try:
    prev_annotations = AnnotationForEpisode.find(
        (AnnotationForEpisode.annotator_id == annotator_id)
    ).all()
except:
    rich.print("[red] Error when fetching previous annotations...")
    prev_annotations = []

for annotation in prev_annotations:
    annotation = cast(AnnotationForEpisode, annotation)
    annotated_episode.add(annotation.episode)


for episode_log_pk in tqdm(episode_log_pks):
    if episode_log_pk in annotated_episode:
        continue
    # get the episode log
    try:
        episode_log = EpisodeLog.get(episode_log_pk)
    except:
        rich.print(
            f"[red] Episode log {episode_log_pk} not found. Skipping..."
        )
        continue
    agent_profiles, messages_and_rewards = episode_log.render_for_humans()
    rich.print(
        f"[b]Welcome {annotator.name}![/b] You are annotating episode {episode_log_pk}"
    )
    layout = Layout()
    layout.split(
        *(
            Layout(f"[green] Agent {idx+1}: {agent_profile}", ratio=1)
            for idx, agent_profile in enumerate(agent_profiles)
        )
    )
    rich.print(layout)
    scores = []
    comments = []

    for idx, turn in enumerate(messages_and_rewards):
        rich.print(f"\n[blue] {turn}\n")
        scores_for_each_agent = []
        for jdx in range(len(agent_profiles)):
            feedback = Prompt.ask(
                f"Do you think the score given to the Agent {jdx+1} is fair? (y/n)",
                choices=["y", "n"],
            )
            scores_for_each_agent.append(1 if feedback == "y" else 0)
        comment = Prompt.ask(
            "[yellow] Any comments for this turn? (Press enter to skip)",
        )
        comments.append(comment)
        scores.append(scores_for_each_agent)
    AnnotationForEpisode(
        episode=episode_log_pk,
        annotator_id=annotator_id,
        scores_for_each_turn=scores,
        comments_for_each_turn=comments,
    ).save()
