"""Tests for local JSON storage backend."""

import os
import tempfile
from pathlib import Path

import pytest
from redis_om.model.model import NotFoundError

# Set environment variable before importing database models
os.environ["SOTOPIA_STORAGE_BACKEND"] = "local"

from sotopia.database import (  # noqa: E402
    AgentProfile,
    EnvironmentProfile,
    EpisodeLog,
    RelationshipProfile,
    RelationshipType,
)
from sotopia.database.storage_backend import LocalJSONBackend  # noqa: E402


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for storage tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override the storage backend to use our temp directory
        backend = LocalJSONBackend(tmpdir)
        # Monkey-patch the get_storage_backend function
        import sotopia.database.storage_backend as sb_module

        original_backend = sb_module._storage_backend
        sb_module._storage_backend = backend

        yield Path(tmpdir)

        # Restore original backend
        sb_module._storage_backend = original_backend


def test_local_backend_save_and_get(temp_storage_dir):
    """Test basic save and get operations."""
    agent = AgentProfile(
        first_name="John",
        last_name="Doe",
        age=30,
        occupation="Engineer",
        gender="Man",
        gender_pronoun="He/Him",
    )

    # Save the agent
    agent.save()
    assert agent.pk is not None

    # Verify file was created
    model_dir = temp_storage_dir / "AgentProfile"
    assert model_dir.exists()
    assert (model_dir / f"{agent.pk}.json").exists()

    # Retrieve the agent
    retrieved = AgentProfile.get(agent.pk)
    assert retrieved.first_name == "John"
    assert retrieved.last_name == "Doe"
    assert retrieved.age == 30
    assert retrieved.occupation == "Engineer"


def test_local_backend_delete(temp_storage_dir):
    """Test delete operation."""
    agent = AgentProfile(
        first_name="Jane",
        last_name="Smith",
        age=25,
        occupation="Doctor",
        gender="Woman",
        gender_pronoun="She/Her",
    )

    agent.save()
    pk = agent.pk
    assert pk is not None

    # Delete the agent
    AgentProfile.delete(pk)

    # Verify file was removed
    model_dir = temp_storage_dir / "AgentProfile"
    assert not (model_dir / f"{pk}.json").exists()

    # Verify get raises NotFoundError
    with pytest.raises(NotFoundError):
        AgentProfile.get(pk)


def test_local_backend_find(temp_storage_dir):
    """Test find operation with filters."""
    # Create multiple agents
    agent1 = AgentProfile(
        first_name="Alice",
        last_name="Johnson",
        age=28,
        occupation="Teacher",
        gender="Woman",
        gender_pronoun="She/Her",
    )
    agent1.save()

    agent2 = AgentProfile(
        first_name="Bob",
        last_name="Williams",
        age=35,
        occupation="Engineer",
        gender="Man",
        gender_pronoun="He/Him",
    )
    agent2.save()

    agent3 = AgentProfile(
        first_name="Charlie",
        last_name="Brown",
        age=28,
        occupation="Designer",
        gender="Man",
        gender_pronoun="He/Him",
    )
    agent3.save()

    # Find by age
    results = AgentProfile.find(AgentProfile.age == 28).all()
    assert len(results) == 2
    assert all(r.age == 28 for r in results)

    # Find by occupation
    results = AgentProfile.find(AgentProfile.occupation == "Engineer").all()
    assert len(results) == 1
    assert results[0].first_name == "Bob"

    # Find by gender
    results = AgentProfile.find(AgentProfile.gender == "Man").all()
    assert len(results) == 2


def test_local_backend_all(temp_storage_dir):
    """Test retrieving all instances."""
    # Create multiple agents
    for i in range(5):
        agent = AgentProfile(
            first_name=f"Agent{i}",
            last_name="Test",
            age=20 + i,
            occupation="Tester",
            gender="Nonbinary",
            gender_pronoun="They/Them",
        )
        agent.save()

    # Retrieve all agents
    all_agents = AgentProfile.all()
    assert len(all_agents) >= 5  # May have more from previous tests


def test_environment_profile_save_and_get(temp_storage_dir):
    """Test environment profile with local storage."""
    env = EnvironmentProfile(
        codename="test_env",
        scenario="Two agents meeting at a coffee shop",
        agent_goals=["Make a new friend", "Have a pleasant conversation"],
        relationship=RelationshipType.stranger,
    )

    env.save()
    assert env.pk is not None

    # Retrieve environment
    retrieved = EnvironmentProfile.get(env.pk)
    assert retrieved.codename == "test_env"
    assert len(retrieved.agent_goals) == 2
    assert retrieved.relationship == RelationshipType.stranger


def test_relationship_profile_save_and_get(temp_storage_dir):
    """Test relationship profile with local storage."""
    # Create two agents first
    agent1 = AgentProfile(
        first_name="Friend1",
        last_name="Test",
        age=30,
        occupation="Engineer",
    )
    agent1.save()

    agent2 = AgentProfile(
        first_name="Friend2",
        last_name="Test",
        age=32,
        occupation="Designer",
    )
    agent2.save()

    # Create relationship
    rel = RelationshipProfile(
        agent_1_id=agent1.pk,  # type: ignore[arg-type]
        agent_2_id=agent2.pk,  # type: ignore[arg-type]
        relationship=RelationshipType.friend,
        background_story="They met in college",
    )
    rel.save()

    # Retrieve relationship
    retrieved = RelationshipProfile.get(rel.pk)  # type: ignore[arg-type]
    assert retrieved.relationship == RelationshipType.friend
    assert retrieved.background_story == "They met in college"


def test_episode_log_save_and_get(temp_storage_dir):
    """Test episode log with local storage."""
    # Create agents and environment first
    agent1 = AgentProfile(first_name="Agent1", last_name="Test")
    agent1.save()

    agent2 = AgentProfile(first_name="Agent2", last_name="Test")
    agent2.save()

    env = EnvironmentProfile(
        codename="test_env",
        scenario="Test scenario",
        agent_goals=["Goal 1", "Goal 2"],
    )
    env.save()

    # Create episode log
    episode = EpisodeLog(
        environment=env.pk,  # type: ignore[arg-type]
        agents=[agent1.pk, agent2.pk],  # type: ignore[list-item]
        tag="test",
        models=["gpt-4", "gpt-4"],
        messages=[
            [
                ("Environment", "Agent1", "Hello Agent1"),
                ("Agent1", "Environment", "Hello!"),
            ]
        ],
        rewards=[7.5, 8.0],
    )
    episode.save()

    # Retrieve episode
    retrieved = EpisodeLog.get(episode.pk)  # type: ignore[arg-type]
    assert retrieved.tag == "test"
    assert len(retrieved.agents) == 2
    assert len(retrieved.models) == 2  # type: ignore[arg-type]


def test_local_backend_pk_generation(temp_storage_dir):
    """Test that primary keys are automatically generated."""
    agent1 = AgentProfile(first_name="Test1", last_name="User")
    agent1.save()

    agent2 = AgentProfile(first_name="Test2", last_name="User")
    agent2.save()

    # PKs should be different UUIDs
    assert agent1.pk != agent2.pk
    assert agent1.pk is not None
    assert agent2.pk is not None


def test_local_backend_update(temp_storage_dir):
    """Test updating an existing record."""
    agent = AgentProfile(
        first_name="Original",
        last_name="Name",
        age=25,
        occupation="Engineer",
    )
    agent.save()
    pk = agent.pk

    # Modify and save again
    agent.first_name = "Updated"
    agent.age = 26
    agent.save()

    # Retrieve and verify
    retrieved = AgentProfile.get(pk)  # type: ignore[arg-type]
    assert retrieved.first_name == "Updated"
    assert retrieved.age == 26
    assert retrieved.pk == pk  # PK should remain the same


def test_local_backend_with_defaults(temp_storage_dir):
    """Test that default values work correctly."""
    # Create agent with minimal fields
    agent = AgentProfile(first_name="Min", last_name="Imal")
    agent.save()

    # Retrieve and check defaults
    retrieved = AgentProfile.get(agent.pk)  # type: ignore[arg-type]
    assert retrieved.age == 0
    assert retrieved.occupation == ""
    assert retrieved.gender == ""
