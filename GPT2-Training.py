# %%
print('''
   _ \                       _| _)  |    |               
  |   | \ \   /  _ \   __|  |    |  __|  __|   _ \   __| 
  |   |  \ \ /   __/  |     __|  |  |    |     __/  |    
 \___/    \_/  \___| _|    _|   _| \__| \__| \___| _|    
                                                         ''')


print('''
   _____ _____ _______ ___     _______        _       _             
  / ____|  __ \__   __|__ \   |__   __|      (_)     (_)            
 | |  __| |__) | | |     ) |_____| |_ __ __ _ _ _ __  _ _ __   __ _ 
 | | |_ |  ___/  | |    / /______| | '__/ _` | | '_ \| | '_ \ / _` |
 | |__| | |      | |   / /_      | | | | (_| | | | | | | | | | (_| |
  \_____|_|      |_|  |____|     |_|_|  \__,_|_|_| |_|_|_| |_|\__, |
                                                               __/ |
                                                              |___/ 
''')

# %%
print('\n')
print('⏳⏳⏳ Starting GPT2 TRAINING by Overfitter ⏳⏳⏳')

# %%
import pandas as pd
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import datetime

# %%
print('✅ --> All imports are done!')

# get own file name
import sys
file_name = sys.argv[0]

print(' --> FILE NAME: ', file_name)

# %%
lyrics_df = pd.read_csv('./datasets/df_songs.csv')
lyrics_df.head()

# %%
print('✅ --> Loaded dataset!')

# %%
test_set = lyrics_df.sample(n=400)
lyrics_df = lyrics_df.loc[~lyrics_df.index.isin(test_set.index)]

# %%
lyrics_dataset = Dataset.from_pandas(lyrics_df)
lyrics_dataset = lyrics_dataset.remove_columns(['Artist', 'Song', 'LyricsWordCount'])

# %%
data_split = lyrics_dataset.train_test_split(test_size=0.10, shuffle=True)

lyris_dataset_train = data_split["train"]
lyris_dataset_valid = data_split["test"]

# %%
#TODO: Create suitable test datset, keep only first 20 words
test_set['True_end_lyrics'] = test_set['Lyrics'].str.split().str[-20:].apply(' '.join)
test_set['Lyrics'] = test_set['Lyrics'].str.split().str[:-20].apply(' '.join)

# %%
print('✅ --> Done test and train split!')

# %% [markdown]
# ---

# %%
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# %%
# NCIHT SICHER OB DAS GOOD PRACTICE IST
tokenizer.pad_token = tokenizer.eos_token

# %%
print('⏳ --> Start tokenizing test and train set')

# %%
dataset_train = lyris_dataset_train.map(lambda examples: tokenizer(examples["Lyrics"], truncation=True))
dataset_valid = lyris_dataset_valid.map(lambda examples: tokenizer(examples["Lyrics"], truncation=True))

# %%
print('✅ --> Done tokenizing test and train set')

# %%
trainings_args = TrainingArguments(
    output_dir="gpt2_train_output",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=1e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=trainings_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_valid,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

# %%
print('⏳ --> Start training-loop!')
trainer.train()
print("✅ --> Training done!\n")

# %%
print('======================================== TRAINING DONE ========================================')
# print date and time
print('Date and time:', datetime.datetime.now())


