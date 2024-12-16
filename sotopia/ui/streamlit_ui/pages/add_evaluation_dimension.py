"""
Definition
@app.get(
    "/evaluation_dimensions/", response_model=dict[str, list[CustomEvaluationDimension]]
)
async def get_evaluation_dimensions() -> dict[str, list[CustomEvaluationDimension]]:
    custom_evaluation_dimensions: dict[str, list[CustomEvaluationDimension]] = {}
    all_custom_evaluation_dimension_list = CustomEvaluationDimensionList.all()
    for custom_evaluation_dimension_list in all_custom_evaluation_dimension_list:
        assert isinstance(
            custom_evaluation_dimension_list, CustomEvaluationDimensionList
        )
        custom_evaluation_dimensions[custom_evaluation_dimension_list.name] = [
            CustomEvaluationDimension.get(pk=pk)
            for pk in custom_evaluation_dimension_list.dimension_pks
        ]
    return custom_evaluation_dimensions

class CustomEvaluationDimensionsWrapper(BaseModel):
    pk: str = ""
    name: str = Field(
        default="", description="The name of the custom evaluation dimension list"
    )
    dimensions: list[CustomEvaluationDimension] = Field(
        default=[], description="The dimensions of the custom evaluation dimension list"
    )
"""

import streamlit as st


def add_evaluation_dimension() -> None:
    st.title("Add Evaluation Dimension")

    with st.expander("Evaluation Dimension List", expanded=True):
        st.text_input("Name", key="name_list")
        st.text_input("Description", key="description_list")
        st.text_input("Evaluation Dimension", key="evaluation_dimension_list")

    with st.expander("Evaluation Dimension", expanded=True):
        st.text_input("Name", key="name_list_item")
        st.text_input("Description", key="description_list_item")
        st.text_input("Evaluation Dimension", key="evaluation_dimension_list_item")
        st.button(
            "Add another Evaluation Dimension", key="add_another_evaluation_dimension"
        )

    st.button("Add Evaluation Dimension", key="add_evaluation_dimension")


add_evaluation_dimension()
