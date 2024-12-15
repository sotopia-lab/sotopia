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
