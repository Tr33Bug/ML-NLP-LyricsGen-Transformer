######## GPT2 TRAINING LOOP ########
#
# Notebook to export the test data for evaluation and train the dataset on the GPT2 model. 
#
####
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
# change dataset

trainData = 'both' # 'rap' 'both'


# %%
print('\n')
print('⏳⏳⏳ Starting GPT2 TRAINING by Overfitter ⏳⏳⏳')

# %%
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup

from tqdm import tqdm, trange
import time
import datetime
import os
import random

# %%
print('✅ --> All imports are done!')

# get own file name
import sys
file_name = sys.argv[0]

print(' --> FILE NAME: ', file_name)

# %%
# set seed for randaomness
seed = 0

#Seeds and hyperparameters
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# set random seed
random.seed(seed)

BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 2e-5
WARMUP_STEPS = 200
EPSILION = 1e-8

# this produces sample output every 100 steps
sample_every = 100

# %% [markdown]
# ---

# %%
# Paths

# top dataset
df_top = pd.read_csv('./datasets/df_top.csv')
df_top  = df_top.drop(columns=['Song', 'Artist'])

# rap dataset
df_rap = pd.read_csv('./datasets/df_rap.csv')
df_rap  = df_rap.drop(columns=['Song', 'Artist'])

# topRap merge dataset
df_songs = pd.read_csv('./datasets/df_songs.csv')
df_songs = df_songs.drop(columns=['Song','LyricsWordCount', 'Artist'])#

# chose dataset on trainData string
if trainData == 'top':
    lyrics_df = df_top
elif trainData == 'rap':
    lyrics_df = df_rap
elif trainData == 'both':
    lyrics_df = df_songs
else:
    print('❌ --> ERROR: trainData string is not correct')



lyrics_df.head()

# %%
print('✅ --> Loaded dataset: ', trainData)

# %% [markdown]
# ---

# %% [markdown]
# ### Export test data

# %%
n = int(len(lyrics_df) * 0.10)
test_samples = lyrics_df.sample(n, random_state=0)
lyrics_df = lyrics_df.drop(test_samples.index)

# %%
test_samples["True_end_lyrics"] = ""
test_samples["Lyrics_Cut"] = ""

for row in test_samples.iterrows():
    lyrics = str(row[1]['Lyrics'])
    lyrics = lyrics.split()
    split = int(len(lyrics) * 0.5)
    
    lyrics_cut = lyrics[:split]
    true_end_lyrics = lyrics[split:]
    
    true_end_lyrics = " ".join(true_end_lyrics)
    lyrics_cut = " ".join(lyrics_cut)
    
    row[1]["True_end_lyrics"] = true_end_lyrics
    row[1]["Lyrics_Cut"] = lyrics_cut

# %%
path_test_samples = './datasets/' + trainData + "_test_samples.csv"
test_samples.to_csv(path_test_samples)
print('✅ --> Exported test sample dataset to : ', path_test_samples)

# %% [markdown]
# ---

# %%
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-medium
configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)

model.resize_token_embeddings(len(tokenizer))

# %%
class GPT2Dataset(Dataset):

    def __init__(self, txt_list, tokenizer, gpt2_type="gpt2", max_length=768):

        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []

        for txt in txt_list:
            encodings_dict = tokenizer('<|startoftext|>' + str(txt) + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

# %%
lyricList = lyrics_df["Lyrics"].tolist()
dataset = GPT2Dataset(lyricList, tokenizer, max_length=768)

# Split into training and validation sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

# %%
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = BATCH_SIZE # Trains with this batch size.
        )

validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = BATCH_SIZE # Evaluate with this batch size.
        )

# %%
optimizer = AdamW(model.parameters(), lr = LEARNING_RATE, eps = EPSILION)

total_steps = len(train_dataloader) * EPOCHS


scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = WARMUP_STEPS, 
                                            num_training_steps = total_steps)

# %%
def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

# %%
print('⏳ --> Start training-loop!')

# %%
total_t0 = time.time()
training_stats = []

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    model = model.cuda()
model = model.to(device)

for epoch_i in range(0, EPOCHS):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, EPOCHS))
    print('Training...\n')
    
    # ========================================
    #               Training
    # ========================================

    t0 = time.time()
    total_train_loss = 0
    model.train()

    print('✅ --> Epoche', epoch_i, 'model.train done!\n')


    for step, batch in enumerate(train_dataloader):

        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)

        model.zero_grad()        
        outputs = model(b_input_ids,labels=b_labels, attention_mask = b_masks, token_type_ids=None)
        loss = outputs[0]  

        batch_loss = loss.item()
        total_train_loss += batch_loss

        # Get sample every x batches.
        if step % sample_every == 0 and not step == 0:

            elapsed = format_time(time.time() - t0)
            print('✅ Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.\n'.format(step, len(train_dataloader), batch_loss, elapsed))
            print('⏳ --> Start evaluating model!\n')
            model.eval()
            print('✅ --> Done evaluating model!\n')
            sample_outputs = model.generate(
                                    bos_token_id=random.randint(1,30000),
                                    do_sample=True,   
                                    top_k=50, 
                                    max_length = 200,
                                    top_p=0.95, 
                                    num_return_sequences=1
                                )
            print('✅ --> Model evaluation done!\n')
            print('⏳ --> Print test-generated text!\n')
            for i, sample_output in enumerate(sample_outputs):
                  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
            
            print('✅ --> Print test-generated text done!\n')
            
            print('⏳ --> Start training model again, in sample Loop\n')
            model.train()
            print('✅ --> Done training model again, in sample Loop\n')
        
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)       
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("✅ --> Training epoch done!\n")
    print("Average training loss: {0:.2f}\n".format(avg_train_loss))
    print("Training epoch took: {:}\n".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================

    print('⏳ --> Start validating model!\n')

    t0 = time.time()
    model.eval()
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)
        
        with torch.no_grad():        

            outputs  = model(b_input_ids, 
#                            token_type_ids=None, 
                             attention_mask = b_masks,
                            labels=b_labels)
          
            loss = outputs[0]  
            
        batch_loss = loss.item()
        total_eval_loss += batch_loss        

    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    validation_time = format_time(time.time() - t0)    

    print("✅ --> Validation epoch done!\n")
    print("Validation Loss: {0:.2f}\n".format(avg_val_loss))
    print("Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("✅ Training complete!\n")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

print('⏳ --> Start saving model!\n')

# %%
# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
output_dir = './model_save/' + trainData + '/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# %%
print('✅ --> Model saved!\n')
print('⏳ --> Start saving training stats!\n')

# %%
pd.set_option('precision', 2)
df_stats = pd.DataFrame(data=training_stats)
df_stats = df_stats.set_index('epoch')
#df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

# export dataframe to csv
df_stats.to_csv('training_stats_' + trainData + '.csv')

# %%
print('✅ --> Training stats saved!\n')

print('======================================== TRAINING DONE ========================================')
# print date and time
print('Date and time:', datetime.datetime.now())


