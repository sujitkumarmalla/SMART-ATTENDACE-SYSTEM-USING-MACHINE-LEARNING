import pickle


with open("faces_data.pkl", "wb") as f:
    pickle.dump([], f)


with open("names.pkl", "wb") as f:
    pickle.dump([], f)

print("All faces and names deleted ✅")
