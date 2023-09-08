from recsys import RatingMatrix, RecSys

rm = RatingMatrix.read_json("raw_ratings.json")
rm.filter(minvotes=50, inplace=True)


model = RecSys(rm)
model.set_testset(0.1)
model.initialize()
model.train(epochs=20)
print(model.validate())
