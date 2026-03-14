import werewolfSource from "../../../../../examples/experimental/werewolves/config.json";

type RawWerewolfConfig = {
    scenario: string;
    description: string;
    role_goals?: Record<string, string>;
    role_secrets?: Record<string, string>;
    state_transition?: Record<string, string>;
    state_properties?: Record<string, unknown>;
    end_conditions?: Array<Record<string, unknown>>;
    agents: Array<{
        name: string;
        role: string;
        team?: string;
    }>;
};

const DEFAULT_AGENT_MODEL = "gpt-4o-mini";
const DEFAULT_EVALUATOR_MODEL = "gpt-4o-mini";
const DEFAULT_MAX_TURNS = 20;
const ENV_ID = "werewolf_env_config";

const rawConfig = werewolfSource as RawWerewolfConfig;

if (!rawConfig.agents?.length) {
    throw new Error("Werewolf configuration must include at least one agent.");
}

const sanitizeId = (name: string, index: number) =>
    `werewolf_${index + 1}_${name.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;

const roleGoals = rawConfig.role_goals ?? {};
const roleSecrets = rawConfig.role_secrets ?? {};

const cleanedDescription =
    rawConfig.description?.replace(/\s+/g, " ").trim() ??
    "Werewolf social deduction scenario.";

const derivedAgents = rawConfig.agents.map((agent, index) => {
    const agentId = sanitizeId(agent.name, index);
    const goal =
        roleGoals[agent.role] ??
        `Advance the objectives of the ${agent.team ?? "village"}.`;
    const secret = roleSecrets[agent.role];

    const publicInfoParts = [
        `${agent.name} is aligned with the ${agent.team ?? "village"}.`,
        `Role: ${agent.role}.`,
        goal ? `Goal: ${goal}` : null,
    ].filter(Boolean);

    return {
        agentId,
        goal,
        profile: {
            pk: agentId,
            agent_id: agentId,
            model_name: DEFAULT_AGENT_MODEL,
            first_name: agent.name,
            last_name: agent.team ?? agent.role,
            occupation: agent.role,
            team: agent.team,
            public_info: publicInfoParts.join(" "),
            secret,
        },
    };
});

export const werewolfAgentIds = derivedAgents.map((entry) => entry.agentId);

export const werewolfAgentProfiles = derivedAgents.map(
    (entry) => entry.profile
);

export const werewolfAgentModels = derivedAgents.map(
    () => DEFAULT_AGENT_MODEL
);

export const werewolfAgentMetadata = rawConfig.agents.map((agent, index) => ({
    id: derivedAgents[index].agentId,
    name: agent.name,
    role: agent.role,
    team: agent.team ?? "Villagers",
}));

export const werewolfPackMetadata = werewolfAgentMetadata
    .filter((agent) => agent.role.toLowerCase() === "werewolf")
    .map((agent) => ({
        id: agent.id,
        displayName: agent.name,
    }));

export const werewolfEnvironmentProfile = {
    pk: ENV_ID,
    env_id: ENV_ID,
    codename: rawConfig.scenario,
    scenario: cleanedDescription,
    agent_goals: derivedAgents.map((entry) => entry.goal),
    model_name: DEFAULT_AGENT_MODEL,
    max_turns: DEFAULT_MAX_TURNS,
    state_transition: rawConfig.state_transition,
    state_properties: rawConfig.state_properties,
    end_conditions: rawConfig.end_conditions,
};

export const werewolfEvaluatorModel = DEFAULT_EVALUATOR_MODEL;
export const werewolfMaxTurns = DEFAULT_MAX_TURNS;
