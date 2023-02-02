import pandas as pd 
import model_park as mp
import log_utils as lu 
import uvicorn 

from fastapi import FastAPI
from pydantic import BaseModel


lu.init_logging('app', 'app_logs.log')
logger = lu.logger

app = FastAPI()


class Tweet(BaseModel):
    keyword: list
    location: list
    text: list


@app.post("/analyze_tweet/", response_model=dict)
def analyze_tweet(tweet: Tweet) -> dict:
    data = pd.DataFrame({'keyword': tweet.keyword, 'location': tweet.location, 'text': tweet.text})
    logger.info(f"Input data: {data}")

    data_handler = mp.DataHandler()
    prepared_df = data_handler.prepare_data(data)
    
    model_keeper = mp.ModelKeeper("Log_Reg_model")
    model = model_keeper.build_model(prepared_df, train=False, do_cv=False)

    answer = {}
    answer['probabilities'] = model.predict_proba(data).tolist()[0]
    answer['class'] = 'real' if model.predict(data) else 'fake'
    logger.info(f"Answer: {answer}")
    return answer

if __name__ == "__main__":
    uvicorn.run("web_server:app", host="127.0.0.1", port=5000, log_level="info")