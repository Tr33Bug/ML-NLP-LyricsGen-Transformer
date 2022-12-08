# ML-NLP-LyricsGen-Transformer
`AI-lab task by Overfitter`

Fine-tuning and prompting a transformer(GPT2) for a song-lyrics generator.
---
Group members and Contributors:
- [Tr33Bug](https://github.com/Tr33Bug)
- [gusse-dev](https://github.com/Gusse-dev)
- [CronJorian](https://github.com/CronJorian)
- [BFuertsch](https://github.com/BenJosh95)



---

## Project structure
0. README.md
1. 10_DataEngineering.ipynb
    - 11_createDataset.py
2.  20_GPT2_TrainingLoop.ipynb
    - 21_GPT2_TrainingLoop.py
    - startTraining.sh
3. 30_ModelEvaluation.ipynb
4. 40_Prompting.ipynb
    - 41_Prompting.py


--- 
## Main project files

### 10_DataEngineering.ipynb
Notebook to generate, clean, and analyze the lyrics dataset for the lyrics generator. 

1. **generate**: to generate the dataset, we use 3 lists from IMDB and the lyricsgenius framework to crawl the song lyrics from the API from (genius.com)[www.genius.com]:
    - Top 49 20th Century: https://www.imdb.com/list/ls058480497/
    - Top 100 AllTime: https://www.imdb.com/list/ls064818015/
    - Top 100 Rapper: https://www.imdb.com/list/ls054191097/
    - LyricsGenius-framework: https://github.com/johnwmillr/LyricsGenius
    
2. **clean**: cleaning the `.txt` files and deleting all unnecessary characters such as `워`, `()`, etc.
3. **analyzing**: viewing graphs, merging the lists of artists, dropping short songs and artists with fewer songs, and counting the most used words. 

In the end, we export the generated dataset files to `df_rap.csv`, `df_songs.csv`, and `df_top.csv`.

### 11_createDataset.py
Python script to generate the datasets from the folders and save the datasets as `df_rap.csv`, `df_songs.csv`, and `df_top.csv`.

### 20_GPT2_TrainingLoop.ipynb
Notebook to export the test data for evaluation and train the dataset on the GPT2 model. 

### 21_GPT2_TrainingLoop.py
Python script exported from `20_GPT2_TrainingLoop.ipynb` file to train remote on the KILab pool PC.  

### startTraining.sh
Helper script to perform the remote training on the KILab pool PC.

### 30_ModelEvaluation.ipynb
Notebook to evaluate the Training of our models and compare them to pretrained GPT2. For that, we load the training results and the models and calculate the BLEU score.

### 40_Prompting.ipynb
Notebook to perform prompting with OpenPrompt on the models. 

### 41_Prompting.py
Python script exported from `40_Prompting.ipynb` file to train remote on the KILab pool PC.

---

