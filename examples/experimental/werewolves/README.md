# Duskmire Werewolves

A text-based social deduction game built on top of `sotopia`. This experimental example demonstrates how to implement complex game phases (Day/Night), roles, and turn-based interactions using the Sotopia framework.

## Overview

In this 6-player game, players are assigned roles (Villager, Werewolf, Seer, Witch) and compete to eliminate the opposing team.

- **Villagers**: Must identify and vote out Werewolves.
- **Werewolves**: Must deceive Villagers and eliminate them at night.
- **Seer**: Can inspect one player's role each night.
- **Witch**: Has one potion to save a victim and one to poison a suspect.

## Features

- **Sequential Discussion**: Utilizes `round-robin` action order during the day, ensuring agents speak one after another and can reference previous arguments.
- **Simultaneous Action**: Night phases and voting are simultaneous to preserve secrecy/fairness.
- **Global Event Notifications**: Players receive system messages about state transitions (e.g., "Entering Night Phase") regardless of their role visibility settings.
- **Safe Elimination**: Role information is hidden from players upon elimination to simulate realistic uncertainty (roles are only revealed in admin logs).

## Running the Game

1. Ensure you have the `sotopia` environment set up.
2. Run the main script:
   ```bash
   python examples/experimental/werewolves/main.py
   ```
   *Note: Ensure your Redis server is running.*

## Configuration

The game is configured via `config.json`. Key settings include:

- **`state_properties`**: Defines the phases (Day/Night).
    - `action_order`: Set to `"round-robin"` for sequential phases (e.g., `Day_discussion`), or `"simultaneous"` for others (e.g., `Day_vote`).
    - `visibility`: Controls who sees messages (`"public"`, `"team"`, `"private"`).
- **`agents`**: Defines the roster and roles.

## Extending

To modify the game logic, check:
- `main.py`: Handles game initialization and elimination logic (`_check_eliminations`).
- `config.json` and `sotopia/envs/social_game.py`: Adjusts game balance, roles, and state transitions.
