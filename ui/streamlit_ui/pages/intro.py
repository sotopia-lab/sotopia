import streamlit as st

st.title("Sotopia Demo App")
st.markdown(
    """
    Sotopia is a place where you can create and explore social interactions between agents.
    You can build your own **agents**, **scenarios**, and **evaluation dimensions**.
    Then try out your ideas and evaluate the results with your custom settings.

    For more interesting projects using Sotopia, check out [our website](https://sotopia.world).
    """
)

st.markdown("### Getting Started")
st.markdown(
    """
    - **Browse Content**: Check out ready-made characters, scenarios, and episodes in their tabs.
    - **Create Your Own**: Add your own characters, scenarios, or evaluation dimensions by clicking the corresponding tab name with a "+" button in the tabs.
    - **Simulate**: Try out simulation by going to the "Simulate" tab and choose your customized settings.
    - **Use Your Database**: If you have a remote Redis database, enter its URL in the sidebar to use your custom data.
    """
)

st.markdown("### API Documentation")
st.markdown(
    """
        For larger scale experiments you may need to use the API instead of the Streamlit UI.
        - The API documentation for current set of Sotopia is [here](https://sotopia-lab--sotopia-fastapi-webapi-serve.modal.run/)
        - When you are hosting your own API, find it in `{YOUR_API_BASE}/docs`.
        - Also see [Sotopia examples](https://github.com/sotopia-lab/sotopia/example) for more information.
    """
)
st.markdown("Current API Base: " + st.session_state.API_BASE)
