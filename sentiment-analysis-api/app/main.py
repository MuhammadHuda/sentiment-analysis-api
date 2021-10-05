from typing import List, Dict

from fastapi import FastAPI
from pydantic import BaseModel

from nlp.SA_DL import SentimentAnalysisDL

class PredictRequest(BaseModel):
    id: List[str]
    text: List[str]
    lang: List[str]
    created_at: List[str]

    class Config:
        schema_extra = {
            "example": {
                "id": [
                       '1295522497282977792',
                       '1296425255800537089'
                      ],
                "text": [
                         'Example tweet one, no sweat',
                         'I think this is supposed to be tweet two'
                        ],
                "lang": [
                         'in',
                         'in'
                        ],
                "created_at": [
                               '2020-08-18T00:46:47',
                               '2020-08-20T12:34:02'
                              ],
            },
        }


class PredictResponse(BaseModel):
    # prediction: List[str]
    prediction: Dict[str, List[str]]

    class Config:
        schema_extra = {
            "example": {
                "prediction": {
                    "id": [
                        '1295522497282977792',
                        '1296425255800537089'
                    ],
                    "text": [
                        'Example tweet one, no sweat',
                        'I think this is supposed to be tweet two'
                    ],
                    "lang": [
                        'in',
                        'in'
                    ],
                    "created_at": [
                        '2020-08-18T00:46:47',
                        '2020-08-20T12:34:02'
                    ],
                    "sentiment_class": [
                        'neu',
                        'neu'
                    ]
                }
            }
        }


app = FastAPI(
    title='Sentiment Analysis API',
    description='API khusus untuk Sentiment Analysis berbasis DL',
    version='0.1')
clf = SentimentAnalysisDL()

tags_metadata = [
    {'name': 'predict',
     'description': 'Predict the sentiment of batches of text',
    }
]

@app.post("/predict", response_model=PredictResponse, tags=['predict'])
def predict(input: PredictRequest = None):
    pred = clf.predict(input.text)
    pr = pred.tolist()
    pr_ = []
    for p in pr:
        if p == 0:
           pr_.append('neu')
        elif p == 1:
            pr_.append('pos')
        else:
            pr_.append('neg')

    new_data = dict()
    new_data['id'] = input.id
    new_data['text'] = input.text
    new_data['lang'] = input.lang
    new_data['created_at'] = input.created_at
    new_data['sentiment_class'] = pr_

    return PredictResponse(prediction = new_data)