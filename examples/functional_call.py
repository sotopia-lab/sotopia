import json

import openai

message = """
Here is the context of this interaction:
Scenario: Two friends hanging out in a Starbuck
Participants: Ethan Smith and Hendrick Heinz
Ethan Smith's background: Ethan Smith is a 29-year-old Male Software Developer.
Hendrick Heinz's background: Hendrick Heinz is a 54-year-old Male Chef.
Ethan Smith's goal: Ask what the friend would want for their birthday without letting them aware (so that they could give a
big surprise to them)
Hendrick Heinz's goal: have a good time with their friend
Turn #1
Ethan Smith said: "Hey Hendrick, I was just thinking about your birthday coming up. Do you have any plans?"
Turn #2
Hendrick Heinz said: "Thanks for remembering, Ethan. I don't have any specific plans yet, but I'm open to suggestions."
Turn #3
Ethan Smith said: "Actually, I was thinking of getting you a birthday present. Any ideas on what you might want?"
Turn #4
Hendrick Heinz said: "No need for a present, Ethan. Just having you here with me is already a great gift."
Turn #5
Ethan Smith [non-verbal communication] smiling and nodding
"""
# functions = [
#     {
#             "name": "reason_about_rewards",
#             "description": "Get the current rating for each participant.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "reasoning": {
#                         "type": "string",
#                         "description": "first reiterate agents' social goals and then reason about what agents say/do and whether that aligns with their goals.",
#                     },
#                     "p1_rate": {
#                         "type": "string",
#                         "description": "rating of participant 1 on achieving their goal(s), on the scale of 0 to 9.",
#                     },
#                     "p2_rate": {
#                         "type": "string",
#                         "description": "rating of participant 2 on achieving their goal(s), on the scale of 0 to 9.",
#                     },
#                 },
#                 "required": ["reasoning", "p1_rate", "p2_rate"],
#             },
#         },
#    {
#             "name": "reason_about_rewards_knowledge",
#             "description": "Get the current rating for each participant from the knowledge perspective. i.e., the persuit of information.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "reasoning": {
#                         "type": "string",
#                         "description": "first analyze how much the participant know about birthday gifts peference of the other participant, and then reason about what agents have said/done and whether the interactions contribute to knowing the birthday gifts peference.",
#                     },
#                     "p1_rate": {
#                         "type": "string",
#                         "description": "rating of participant 1, on the scale of 0 to 9.",
#                     },
#                 },
#                 "required": ["reasoning", "p1_rate"],
#             },
#         },
#     {
#             "name": "reason_about_rewards_secret",
#             "description": "Get the current rating for each participant from the secret perspective. i.e., the concealment of information/intention.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "reasoning": {
#                         "type": "string",
#                         "description": "first analyze how much the second participant is aware of first participants' plan of buing them birthday gift, and then reason about what agents have said/done and whether the interactions leak the intentions.",
#                     },
#                     "p1_rate": {
#                         "type": "string",
#                         "description": "rating of participant 1, on the scale of -9 to 0. (negative values indicate the participant leak their intentions)",
#                     },
#                 },
#                 "required": ["reasoning", "p1_rate"],
#             },
#         },
#         {    "name": "reason_about_rewards_relationship",
#             "description": "Get the current rating for each participant from the relationship perspective. i.e., the relationship between the two participants.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "reasoning": {
#                         "type": "string",
#                         "description": "first analyze how much the two participants maintain their relationship, and then reason about what agents have said/done and whether the interactions contribute to the relationship.",
#                     },
#                     "p1_rate": {
#                         "type": "string",
#                         "description": "rating of participant 1, on the scale of -5 to 5. (negative values indicate the participant damage the relationship while positive values indicate the participant contribute the relationship)",
#                     },
#                     "p2_rate": {
#                         "type": "string",
#                         "description": "rating of participant 2, on the scale of -5 to 5. (negative values indicate the participant damage the relationship while positive values indicate the participant contribute the relationship)",
#                     },
#                 },
#                 "required": ["reasoning", "p1_rate", "p2_rate"],
#             },
#         },
# ]


# Step 1: send the conversation and available functions to GPT
# for f in functions:
#     messages = [{"role": "user", "content": message}]
#     response = openai.ChatCompletion.create(
#         model="gpt-4-0613",
#         messages=messages,
#         # functions=functions,
#         # function_call={"name": f["name"]},  # auto is default, but we'll be explicit
#     )
#     print(response["choices"][0]["message"]["function_call"]["arguments"])
