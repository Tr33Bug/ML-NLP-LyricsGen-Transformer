# ML-NLP-LyricsGen-Transformer
`AI-lab task by Overfitter`

Fine-tuning and prompting a transformer(GPT2) for a song-lyrics generator.
---
Groupmembers and Contributors:
- [Tr33Bug](https://github.com/Tr33Bug)
- [gusse-dev](https://github.com/Gusse-dev)
- [CronJorian](https://github.com/CronJorian)
- [BFuertsch](https://github.com/BenJosh95)



---

## Projectstructure
0. README.md
1. DataEngineering.ipynb
    - createDataset.py
2.  GPT2_TrainingLoop.ipynb
    - GPT2_TrainingLoop.py
    - startTraining.sh
3. ModelEvaluation.ipynb



### dataEngineering.ipynb

The Dataengineering takes part in the **DataEngineering.ipynb**. In this Notebook, we generate and clean the datasets we need to perform the NLP-traning. For that we first make a list of the top 100 artists of all the time and of the top 100 rapper. The list ressources are: 
- Top 49 20th Century: https://www.imdb.com/list/ls058480497/
- Top 100 AllTime: https://www.imdb.com/list/ls064818015/
- Top Rapper: https://www.imdb.com/list/ls054191097/

After that, we merge the two lists, to get 3 lists in total:
1. Top Artists100 (top)
2. Top Rap-Artists (rap)
3. the combination of both (both)

With these lists, we crawl the top 50 song lyrics from every artists on the list and save them to lyrics folders. For that, we use the genius.com api with the python framework https://github.com/johnwmillr/LyricsGenius. 
To use the crawler, create a .txt file in the project folder named `geniusToken.txt` and past the token from the genius.com api in it.  

With that dataset of ca. 8000 song lyrics, we clean the files, analyse them and write them to three csv tables. 

#### createDataset.py

For the cleaning and creating of the csv files (df_top.csv, df_rap.csv and df_songs.csv) you can also use the **createDataset.py** script.

