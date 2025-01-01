from redis_om import JsonModel

class myModel(JsonModel):
    name: str

if __name__ == "__main__":
    model = myModel(name="test")
    model.save()
    print(myModel.get(model.pk).name)
    model.delete()