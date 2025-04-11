class SotopiaDimensions(BaseModel):
    deal_made: tuple[str, int] = Field(
        ...,
        description="Please provide a comprehensive analysis on whether the agents have reached an agreement (Hint: pay more attention to the last few rounds to determine whether they have made an agreement). Remember a verbal agreement is sufficient and necessary. In the 'reasoning' field, provide a comprehensive account of the logic or thought process that led you to your conclusion. Further, provide an integer score in [0, 1] the 'score' field. 0 represents no agreement, 1 represents agreement.",
    )
    
    point: tuple[str, int] = Field(
        ...,
        description="Please first reiterate the rubics for the point evaluation, then provide a comprehensive analysis on agent's performance measured by points. In 'reasoning' field you should first find out if the agents are willing to go ahead with the current offer (CHECK YOUR EVALUATION IN DEAL_MADE DIMENSION), if your answer is yes (i.e. your score is 1), then use the rubics presented in corresponding agents' goals to determine the points they got (USE LINEAR COMBINATION OF THE NEAREST LEVEL if there are no matching level). If no (i.e. your score is 0) then first list the score levels and use the averaged score of all levels IN CURRENT NEGOTIATION (NOT THE NEXT ONE). In 'score' field, provide your calculated average points. [Example] In the conversation the candidate is not satisfied with the offer of $150000, then the points should **NOT** be based on this number, but **the averaged score of all levels** in the current negotiation. Also be careful that there are multiple dimensions and you have to get the average separately and then add them up.",
    )
    
    transactivity: tuple[str, int] = Field(
        ...,
        description="Analyze the provided social interaction episode between the given pair/team, focusing on identifying instances of transactive exchanges. Evaluate the level of transactivity by considering the following aspects: elaboration, building upon ideas, questioning, argumentation. Analyze whether these transactive patterns persist consistently across the entire interaction or if there are notable variations throughout the exchange. In the 'reasoning' field, provide a comprehensive account of the logic and thought process that led to your conclusion. Consider how the observed instances of transactivity contribute to or detract from the overall quality and depth of the interaction. In the 'score' field, provide an integer score ranging from 0 to 10, where a higher score indicates a higher level of transactivity."
    )
    
    verbal_equity: tuple[str, int] = Field(
        ...,
        description="Analyze the script and measure the level of verbal equity reflected in the interaction between the agents. And then analyze the extent to which the interaction shows a balanced distribution of speaking opportunities among team members. In the 'reasoning' field, provide a comprehensive account of the logic or thought process that led you to your conclusion. Further, provide an integer score ranging from 0 and 10 in the 'score' field. A higher score indicates a higher level of verbal equity."
    )

    
    @validator("point", allow_reuse=True)
    def int_validator(cls, v: tuple[str, int]) -> tuple[str, int]:
        assert isinstance(v[1], int)
        return v
    @validator("deal_made", allow_reuse=True)
    def zero_or_one_validator(cls, v: tuple[str, int]) -> tuple[str, int]:
        assert v[1] == 0 or v[1] == 1
        return v
    
    @validator("transactivity", "verbal_equity", allow_reuse=True)
    def zero_to_ten_validator(cls, v: tuple[str, int]) -> tuple[str, int]:
        assert v[1] >= 0 and v[1] <= 10
        return v
    
class JobNegotiationDimensions(BaseModel):
    # Satisfaction questions
    satisfaction_1: tuple[str, int] = Field(
        ...,
        description="You are asked to complete a survey question: How satisfied are you with your overall experience during this interaction? Please first share your thoughts on why you rated this way, and then rate your level of agreement on a scale from 1 to 5, in which 1: Strongly Disagree, 2: Disagree, 3: Neutral, 4: Agree, 5: Strongly Agree."
    )
    satisfaction_2: tuple[str, int] = Field(
        ...,
        description="You are asked to complete a survey question: How satisfied are you with your interlocutor's performance during this interaction? Please first share your thoughts on why you rated this way, and then rate your level of agreement on a scale from 1 to 5, in which 1: Strongly Disagree, 2: Disagree, 3: Neutral, 4: Agree, 5: Strongly Agree."
    )
    satisfaction_3: tuple[str, int] = Field(
        ...,
        description="You are asked to complete a survey question: How satisfied are you with your own performance during this interaction? Please first share your thoughts on why you rated this way, and then rate your level of agreement on a scale from 1 to 5, in which 1: Strongly Disagree, 2: Disagree, 3: Neutral, 4: Agree, 5: Strongly Agree."
    )

    # Effort / Frustration questions
    effort_score: tuple[str, int] = Field(
        ...,
        description="You are asked to complete a survey question: How hard did you have to work to accomplish your level of performance during the interaction? Please first share your thoughts on why you rated this way, and then rate your level of agreement on a scale from 1 to 5, in which 1: Very Little Effort, 2: Little Effort, 3: Moderate Effort, 4: Much Effort, 5: Extreme Effort."
    )
    frustration_score: tuple[str, int] = Field(
        ...,
        description="You are asked to complete a survey question: How insecure, discouraged, irritated, stressed, and annoyed were you during the interaction? Please first share your thoughts on why you rated this way, and then rate your level of agreement on a scale from 1 to 5, in which 1: Not at all, 2: A little, 3: Moderately, 4: Quite a bit, 5: Extremely."
    )

    # Personal perception questions
    trustworthy_score: tuple[str, int] = Field(
        ...,
        description="You are asked to complete a survey question: I believe that the interlocutor is trustworthy. Please first share your thoughts on why you rated this way, and then rate your level of agreement on a scale from 1 to 5, in which 1: Strongly Disagree, 2: Disagree, 3: Neutral, 4: Agree, 5: Strongly Agree."
    )
    honest_score: tuple[str, int] = Field(
        ...,
        description="You are asked to complete a survey question: I believe that the interlocutor is honest. Please first share your thoughts on why you rated this way, and then rate your level of agreement on a scale from 1 to 5, in which 1: Strongly Disagree, 2: Disagree, 3: Neutral, 4: Agree, 5: Strongly Agree."
    )
    dependable_score: tuple[str, int] = Field(
        ...,
        description="You are asked to complete a survey question: I believe that the interlocutor is dependable. Please first share your thoughts on why you rated this way, and then rate your level of agreement on a scale from 1 to 5, in which 1: Strongly Disagree, 2: Disagree, 3: Neutral, 4: Agree, 5: Strongly Agree."
    )
    reliable_score: tuple[str, int] = Field(
        ...,
        description="You are asked to complete a survey question: I believe that the interlocutor is reliable. Please first share your thoughts on why you rated this way, and then rate your level of agreement on a scale from 1 to 5, in which 1: Strongly Disagree, 2: Disagree, 3: Neutral, 4: Agree, 5: Strongly Agree."
    )

    @validator("satisfaction_1", "satisfaction_2", "satisfaction_3", 
               "effort_score", "frustration_score", 
               "trustworthy_score", "honest_score", "dependable_score", "reliable_score", 
               allow_reuse=True)
    def one_to_five_validator(cls, v: tuple[str, int]) -> tuple[str, int]:
        assert v[1] >= 1 and v[1] <= 5
        return v