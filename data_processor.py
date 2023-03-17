import pandas as pd
import numpy as np

#Final Dataset - definition
# - text -> text value
# - class -> 1 if inappropriate 0 if not

dataset = pd.DataFrame(columns=["text", "class"])

#HSAO Dataset - preprocessing
#Cleaning Steps
#   - Remove ! in beggining of sentence
#   - Remove ReTweet RT and @person:

hsaol = pd.read_csv("downloads/hsaol.csv", sep=",")
hsaol = hsaol[["tweet","class"]]

hsaol["tweet"] = hsaol["tweet"].str.replace("^!*", "")
hsaol["tweet"] = hsaol["tweet"].str.replace("\sRT\s@.*:", " ")
hsaol["inappro"] = (hsaol["class"] < 2).astype(int)

hsaol = hsaol[["tweet", "inappro"]].rename(columns={'tweet': 'text', 'inappro': 'class'})
dataset = pd.concat([dataset, hsaol], ignore_index=True)

#MHS Dataset - preprocessing

measuringhatespeech = pd.read_csv("downloads/measuring_hate_speech.csv", sep=",")[["text", "hate_speech_score"]]

measuringhatespeech["inappro"] = (measuringhatespeech["hate_speech_score"] > 0.5).astype(int)

measuringhatespeech = measuringhatespeech[["text", "inappro"]].rename(columns={'inappro': 'class'})
dataset = pd.concat([dataset, measuringhatespeech], ignore_index=True)


#Display dataset info
print(f"Length: {len(dataset)}")
print(dataset["class"].value_counts())

train, validate, test = np.split(dataset.sample(frac=1), [int(.7*len(dataset)), int(.9*len(dataset))])


print(f"Train: {len(train)} {int(len(train)/len(dataset)*100)}%")
print(f"Validate: {len(validate)} {int(len(validate)/len(dataset)*100)}%")
print(f"Test: {len(test)} {int(len(test)/len(dataset)*100)}%")

train.to_csv("data/train.csv")
validate.to_csv("data/validate.csv")
test.to_csv("data/test.csv")
dataset.to_csv("data/full.csv")
