import streamlit as st

from sotopia.database import CustomEvaluationDimension

from ui.streamlit_ui.rendering import (
    get_evaluation_dimensions,
    render_evaluation_dimension,
    get_distinct_evaluation_dimensions,
    render_evaluation_dimension_list,
)

st.title("Evaluation Dimensions")

st.write("Here are some instructions about using the evaluation dimension renderer.")


def display_evaluation_dimensions() -> None:
    # st.title("Evaluation Dimensions")
    distinct_dimensions: list[CustomEvaluationDimension] = (
        get_distinct_evaluation_dimensions()
    )

    # sort the dimensions by name
    distinct_dimensions.sort(key=lambda x: x.name)

    with st.expander("Evaluation Dimensions", expanded=True):
        all_dimensions = []
        col1, col2 = st.columns(2, gap="medium")
        for i, dimension in enumerate(distinct_dimensions):
            with col1 if i % 2 == 0 else col2:
                render_evaluation_dimension(dimension)

    with st.expander("Evaluation Dimension Lists", expanded=True):
        all_dimension_lists: dict[str, list[CustomEvaluationDimension]] = (
            get_evaluation_dimensions()
        )
        col1, col2 = st.columns(2, gap="medium")
        for i, (dimension_list_name, dimensions) in enumerate(
            all_dimension_lists.items()
        ):
            all_dimensions: list[CustomEvaluationDimension] = [
                CustomEvaluationDimension(**dimension) for dimension in dimensions
            ]
            with col1 if i % 2 == 0 else col2:
                render_evaluation_dimension_list(dimension_list_name, all_dimensions)


display_evaluation_dimensions()
