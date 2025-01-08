# isort: skip_file
# ruff: noqa: E402
import streamlit as st
from ui.rendering import (
    render_environment_profile,
    local_css,
    get_scenarios,
)
from sotopia.database import BaseEnvironmentProfile
from redis import Redis, ConnectionError, AuthenticationError


local_css("./css/style.css")


def verify_redis_connection(url: str) -> tuple[bool, str]:
    """Verify Redis connection and return status."""
    try:
        # Parse URL components for direct connection test
        if "@" in url:
            _, credentials_and_host = url.split("//")
            password, host_and_port = credentials_and_host.split("@")
            password = password.split(":")[1]
            host, port = host_and_port.split(":")
        else:
            host = "localhost"
            port = "6379"
            password = None

        # Test connection
        redis_client = Redis(
            host=host, port=int(port), password=password, socket_timeout=5
        )
        redis_client.ping()
        return True, "Connection successful"
    except ConnectionError:
        return (
            False,
            f"Could not connect to Redis at {host}:{port}. Please verify the server is running and accessible.",
        )
    except AuthenticationError:
        return False, "Authentication failed. Please verify the password."
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def display_scenarios() -> None:
    # Set Redis connection URL
    # connection_successful, message = verify_redis_connection(redis_url)

    # if not connection_successful:
    #     st.error(f"Redis Connection Error: {message}")
    #     print(f"Redis Connection Error: {message}************************")
    #     st.stop()
    # else:
    # st.success(f"Redis Connection Successful: {message}")
    #    print(f"Redis Connection Successful: {message}************************")

    st.title("Scenarios")
    scenarios = get_scenarios()
    try:
        col1, col2 = st.columns(2, gap="medium")
        for index, (codename, scenario) in enumerate(scenarios.items()):
            with col1 if index % 2 == 0 else col2:
                environment_profile = BaseEnvironmentProfile(**scenario)
                render_environment_profile(environment_profile)
                st.write("---")
    except Exception as e:
        print(f"Error getting scenarios: {e}")
        st.error(
            "Failed to retrieve scenarios. Please check your Redis connection and try again."
        )


display_scenarios()
