import streamlit as st
import requests
from sotopia.database import BaseCustomEvaluationDimension
from ui.rendering.get_elements import (
    get_distinct_evaluation_dimensions,
)


def add_evaluation_dimension() -> None:
    st.title("Add Evaluation Dimensions")

    # Initialize session state for dimensions list if it doesn't exist
    if "dimensions" not in st.session_state:
        st.session_state.dimensions = []

    # Get existing dimensions
    existing_dimensions = get_distinct_evaluation_dimensions()

    # Tab-based interface for adding dimensions
    tab1, tab2 = st.tabs(["Add New Dimension", "Select Existing Dimension"])

    # Tab 1: Add New Dimension
    with tab1:
        with st.form(key="add_dimension_form"):
            st.subheader("Add New Dimension")
            dim_name = st.text_input("Dimension Name")
            dim_description = st.text_area("Dimension Description")
            col1, col2 = st.columns(2)
            with col1:
                range_low = st.number_input("Range Low", value=0)
            with col2:
                range_high = st.number_input("Range High", value=10)

            add_dimension = st.form_submit_button("Add Dimension")

            if add_dimension and dim_name and dim_description:
                new_dimension = BaseCustomEvaluationDimension(
                    name=dim_name,
                    description=dim_description,
                    range_low=range_low,
                    range_high=range_high,
                )
                st.session_state.dimensions.append(new_dimension)
                st.success(f"Added dimension: {dim_name}")
                st.rerun()

    # Tab 2: Select Existing Dimension
    with tab2:
        with st.form(key="select_dimension_form"):
            st.subheader("Select Existing Dimension")

            # Create a list of dimension names for the selectbox
            dimension_options = [
                f"{dim.name} (Range: [{dim.range_low}, {dim.range_high}])"
                for dim in existing_dimensions
            ]

            if dimension_options:
                selected_dimension = st.selectbox(
                    "Choose a dimension",
                    options=dimension_options,
                    format_func=lambda x: x.split(" (Range")[
                        0
                    ],  # Show only the name in the dropdown
                )

                # Show details of selected dimension
                if selected_dimension:
                    selected_idx = dimension_options.index(selected_dimension)
                    dim = existing_dimensions[selected_idx]
                    st.info(f"Description: {dim.description}")

                add_existing = st.form_submit_button("Add Selected Dimension")

                if add_existing:
                    selected_idx = dimension_options.index(selected_dimension)
                    dim_to_add = existing_dimensions[selected_idx]

                    # Check if dimension already exists in current list
                    if any(
                        d.name == dim_to_add.name for d in st.session_state.dimensions
                    ):
                        st.error(
                            f"Dimension '{dim_to_add.name}' already exists in the list"
                        )
                    else:
                        st.session_state.dimensions.append(dim_to_add)
                        st.success(f"Added existing dimension: {dim_to_add.name}")
                        st.rerun()
            else:
                st.warning("No existing dimensions available")

    # Display current dimensions with delete buttons
    if st.session_state.dimensions:
        st.subheader("Current Dimensions")
        for idx, dim in enumerate(st.session_state.dimensions):
            col1, col2, col3 = st.columns([3, 1, 0.5])
            with col1:
                with st.expander(f"Dimension {idx + 1}: {dim.name}", expanded=False):
                    st.write(f"Description: {dim.description}")
                    st.write(f"Range: {dim.range_low} - {dim.range_high}")
            with col2:
                if st.button("Delete", key=f"delete_{idx}"):
                    st.session_state.dimensions.pop(idx)
                    st.success(f"Deleted dimension: {dim.name}")
                    st.rerun()
            with col3:
                st.write("")  # Spacer

        # Submit form at the bottom
        with st.form(key="submit_form"):
            st.subheader("Submit Evaluation Dimension List")
            list_name = st.text_input("List Name")

            col1, col2 = st.columns(2)
            with col1:
                submit_button = st.form_submit_button("Submit All")
            with col2:
                clear_button = st.form_submit_button("Clear All")

            if submit_button and list_name:
                try:
                    wrapper = BaseCustomEvaluationDimension(
                        name=list_name, dimensions=st.session_state.dimensions
                    )
                    # st.write(wrapper.dict())

                    response = requests.post(
                        f"{st.session_state.API_BASE}/evaluation_dimensions",
                        json=wrapper.dict(),
                    )

                    if response.status_code == 200:
                        st.success("Successfully created evaluation dimension list!")
                        # st.session_state.dimensions = []
                        # st.rerun()
                    else:
                        st.error(f"Error: {response.json()['detail']}")

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

            if clear_button:
                st.session_state.dimensions = []
                st.rerun()


add_evaluation_dimension()
