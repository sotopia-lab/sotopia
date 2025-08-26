"""
Constants for multi-agent test scenarios.
"""

# Redis configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_URL = f"redis://:{REDIS_HOST}:{REDIS_PORT}"

# Model configurations
STANDARD_MODELS = {
    "env": "gpt-4o",
    "agent1": "gpt-4o",
    "agent2": "gpt-4o",
    "agent3": "gpt-4o",
}

FAST_MODELS = {
    "env": "gpt-4o-mini",
    "agent1": "gpt-4o-mini",
    "agent2": "gpt-4o-mini",
    "agent3": "gpt-4o-mini",
}

# Multi-agent test scenarios
AUCTION_SCENARIO = {
    "name": "Three-Party Auction",
    "description": "Antique auction negotiation: A seller has a rare vintage item. Two buyers are competing to purchase it.",
    "agents": ["seller", "buyer1", "buyer2"],
    "dynamics": "competitive_bidding",
}

GROUP_DECISION_SCENARIO = {
    "name": "Group Decision Making",
    "description": "A team of colleagues must decide on a restaurant for their company dinner.",
    "agents": ["organizer", "vegetarian", "budget_conscious", "foodie"],
    "dynamics": "consensus_building",
}

RESOURCE_SHARING_SCENARIO = {
    "name": "Resource Allocation",
    "description": "Multiple departments compete for limited company resources and must negotiate fair distribution.",
    "agents": ["hr_manager", "tech_lead", "marketing_director", "finance_head"],
    "dynamics": "resource_competition",
}
