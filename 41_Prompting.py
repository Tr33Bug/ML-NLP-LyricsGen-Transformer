######## PROMPTING SCRIPT ########
#
# Python script exported from `40_Prompting.ipynb` file to train remote on the KILab pool PC.
#
####

# %% [markdown]
# One of the few resources found to Prefix Templates with OpenPrompt
# 
# > https://github.com/thunlp/OpenPrompt/blob/main/tutorial/2.1_conditional_generation.py

# %% [markdown]
# ### Imports

# %%
from openprompt import PromptDataLoader, PromptForGeneration
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import PrefixTuningTemplate
from openprompt.utils.metrics import generation_metric
from sklearn.model_selection import train_test_split
from datasets.dataset_dict import DatasetDict, Dataset
# from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup
import torch
from torch.optim import AdamW
import pandas as pd
from pathlib import Path


# %% [markdown]
# ### Variables

# %%
base_path = "./"
csv_data = "./datasets/df_songs.csv"
used_model = "gpt2"

train_split = 0.7
epochs = 7
batch_size = 2

# %% [markdown]
# ### Model / Data Preparation

# %% [markdown]
# #### Dataset

# %% [markdown]
# Read the CSV, remove everything except the lyrics. Add index for flavour.

# %%
lyrics_df = pd.read_csv(csv_data)
lyrics_df = lyrics_df.drop(
    columns=["Artist", "Song", "LyricsWordCount"], errors="ignore"
).reset_index(level=0)


# %% [markdown]
# Split the dataset and create an DatasetDict

# %%
train_df, validation_df = train_test_split(lyrics_df, train_size=train_split)
train_dataset, validation_dataset = Dataset.from_pandas(train_df), Dataset.from_pandas(
    validation_df
)
raw_dataset = DatasetDict({"train": train_dataset, "validation": validation_dataset})


# %% [markdown]
# Create a new dataset with a mapped `InputExample` for each sample

# %%
dataset = {}
for split in ['train', 'validation']:
    dataset[split] = []
    for data in raw_dataset[split]:
        # input_example = InputExample(text_a = data['premise'], text_b = data['hypothesis'], label=int(data['label']), guid=data['idx'])
        input_example = InputExample(text_a = data['Lyrics'], guid=data['index'])
        dataset[split].append(input_example)

# %% [markdown]
# Steal this dataloader wrapper function 🐱‍👤

# %%
def get_dataloader(
    dataset_split, template, tokenizer, wrapper_class, shuffle=False, batch_size=32
):
    """Returns a prompt data load for a given dataset split and template"""

    return PromptDataLoader(
        dataset=dataset_split,
        template=template,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=wrapper_class,
        max_seq_length=256,
        decoder_max_length=256,
        batch_size=batch_size,
        shuffle=shuffle,
        teacher_forcing=False,
        predict_eos_token=True,
        truncate_method="head",
    )


# %% [markdown]
# #### Model (PLM)

# %%
plm, tokenizer, model_config, WrapperClass = load_plm(used_model, used_model)


# %%
# # tokenizer = GPT2Tokenizer.from_pretrained(used_model, bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-medium

# tokenizer.bos_token = "<|startoftext|>"
# tokenizer.eos_token = "<|endoftext|>"
# tokenizer.pad_token = "<|pad|>"


# %% [markdown]
# ### Prompt-Based Fine-Tuning

# %% [markdown]
# Create a template.
# The used template (line 1) equals the last template (line 7), so that the text param can be omitted.

# %%
template = PrefixTuningTemplate(model=plm, tokenizer=tokenizer, text=' {"placeholder":"text_a"} {"special": "<eos>"} {"mask"} ', using_decoder_past_key_values=False)

# Are the tokens necessary? Probably not:
# # You may observe that the example doesn't end with <|endoftext|> token. Don't worry, adding specific end-of-text token
# # is a language-model-specific token. we will add it for you in the TokenizerWrapper once you pass `predict_eos_token=True`

# template = PrefixTuningTemplate(model=plm, tokenizer=tokenizer, text='<|startoftext|>{"placeholder":"text_a"} {"mask"}<|endoftext|>')

# %% [markdown]
# Create one example and print it, to see what it looks like.

# %%
print(template.wrap_one_example(dataset['train'][0]))

# %%
train_dataloader = get_dataloader(
    dataset["train"],
    template,
    tokenizer,
    WrapperClass,
    shuffle=True,
    batch_size=batch_size,
)
validation_dataloader = get_dataloader(
    dataset["validation"],
    template,
    tokenizer,
    WrapperClass,
    shuffle=False,
    batch_size=batch_size,
)


# %%
prompt_model = PromptForGeneration(plm=plm,template=template, freeze_plm=True,tokenizer=tokenizer)

# %%
# Follow PrefixTuning（https://github.com/XiangLi1999/PrefixTuning), we also fix the language model
# only include the template's parameters in training.

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [
            p
            for n, p in template.named_parameters()
            if (not any(nd in n for nd in no_decay)) and p.requires_grad
        ],
        "weight_decay": 0.0,
    },
    {
        "params": [
            p
            for n, p in template.named_parameters()
            if any(nd in n for nd in no_decay) and p.requires_grad
        ],
        "weight_decay": 0.0,
    },
]

optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, eps=1e-8)


# %%
tot_step  = len(train_dataloader)*5
scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)

# %%
generation_arguments = {
    "max_length": 512,
    "max_new_tokens": None,
    "min_length": 5,
    "temperature": 1.0,
    "do_sample": False,
    "top_k": 0,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "num_beams": 5,
    "bad_words_ids": [[628], [198]]
}

def evaluate(prompt_model, dataloader):
    generated_sentence = []
    groundtruth_sentence = []
    prompt_model.eval()

    for step, inputs in enumerate(dataloader):
        _, output_sentence = prompt_model.generate(inputs, **generation_arguments)
        generated_sentence.extend(output_sentence)
        groundtruth_sentence.extend(inputs['tgt_text'])
    score = generation_metric(generated_sentence, groundtruth_sentence, "sentence_bleu")
    print("test_score", score, flush=True)
    return generated_sentence

# %%
# training and generation.
global_step = 0
tot_loss = 0
log_loss = 0
for epoch in range(epochs):
    prompt_model.train()
    for step, inputs in tqdm(enumerate(train_dataloader)):
        global_step +=1
        loss = prompt_model(inputs)
        loss.backward()
        tot_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(template.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if global_step %500 ==0:
            print("Epoch {}, global_step {} average loss: {} lr: {}".format(epoch, global_step, (tot_loss-log_loss)/500, scheduler.get_last_lr()[0]), flush=True)
            log_loss = tot_loss

generated_sentence = evaluate(prompt_model, validation_dataloader)

# %%
with open(base_path + "generated_sentences.txt",'w') as f:
    for i in generated_sentence:
        f.write(i+"\n")


