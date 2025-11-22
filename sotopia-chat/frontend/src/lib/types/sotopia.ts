export type RelationshipType = 0 | 1 | 2 | 3 | 4 | 5;

export interface AgentProfile {
    pk?: string;
    first_name: string;
    last_name: string;
    age?: number;
    occupation?: string;
    gender?: string;
    gender_pronoun?: string;
    public_info?: string;
    big_five?: string;
    moral_values?: string[];
    schwartz_personal_values?: string[];
    personality_and_values?: string;
    decision_making_style?: string;
    secret?: string;
    model_id?: string;
    [k: string]: unknown;
}

export interface EnvironmentProfile {
    pk?: string;
    codename?: string;
    source?: string;
    scenario?: string;
    agent_goals?: string[];
    relationship?: RelationshipType;
    age_constraint?: string;
    occupation_constraint?: string;
    agent_constraint?: string[][];
    [k: string]: unknown;
}

export interface EnvironmentList {
    pk?: string;
    name: string;
    environments: string[];
    agent_index?: string[];
}

export interface EnvAgentComboStorage {
    pk?: string;
    env_id?: string;
    agent_ids?: string[];
    [k: string]: unknown;
}

export interface MessageTransaction {
    pk?: string;
    timestamp_str: string;
    sender: string;
    message: string;
    [k: string]: unknown;
}

export interface SessionTransaction {
    pk?: string;
    expire_time?: number;
    session_id: string;
    client_id: string;
    server_id: string;
    client_action_lock?: string;
    message_list: MessageTransaction[];
    [k: string]: unknown;
}

export interface EpisodeLog {
    pk?: string;
    environment: string;
    agents: string[];
    tag?: string;
    models?: string[];
    messages: [string, string, string][][];
    reasoning: string;
    rewards: unknown[];
    rewards_prompt: string;
    [k: string]: unknown;
}
