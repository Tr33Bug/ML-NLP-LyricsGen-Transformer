######## GUI SCRIPT ########
#
# Python script to demonstrate the song lyrics generation via a GUI.
#
####

import tkinter
import customtkinter
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import pipeline
import threading

customtkinter.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

# tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2') 

# paths
root = './'
driveFolder = root + 'trainingRuns/TrainRuns-DriveSync/'
runFolder = '22_12_05_Run_05_E15_BS_2_S_7_TopRapBoth/'
trainingBothCsv = driveFolder + runFolder + 'training_stats_both.csv'
trainingRapCsv = driveFolder + runFolder + 'training_stats_rap.csv'
trainingTopCsv = driveFolder + runFolder + 'training_stats_top.csv'
savedModelBoth = driveFolder + runFolder + 'model_save/both/'
savedModelRap = driveFolder + runFolder + 'model_save/rap/'
savedModelTop = driveFolder + runFolder + 'model_save/top/'
testLyricsBoth = root + 'datasets/both_test_samples.csv'
testLyricsRap = root + 'datasets/rap_test_samples.csv'
testLyricsTop = root + 'datasets/top_test_samples.csv'

# define ctkinter window
app = customtkinter.CTk()
app.geometry("560x700")
app.title("Song Lyrics Generator")

# generate song lyrics from given parameters
def button_callback():
    match radiobutton_var.get():
        case "rap":
            # load model from file
            model_rap = GPT2LMHeadModel.from_pretrained(savedModelRap)
            model_rap.resize_token_embeddings(len(tokenizer))

            # pipline for text generation
            lyrics_pipeline_rap = pipeline('text-generation',model=model_rap, tokenizer=tokenizer)
            song_input = text_prompt.get("1.0", 'end-1c') + '\n'
            
            result = lyrics_pipeline_rap(song_input, min_length=int(entry_min.get()), max_length=int(entry_max.get()))[0]['generated_text']

            text_lyrics.delete('1.0', 'end') # delete last lyrics
            text_lyrics.insert("0.0", result) # insert generated lyrics
        case "top":
            # load model from file
            model_top = GPT2LMHeadModel.from_pretrained(savedModelTop)
            model_top.resize_token_embeddings(len(tokenizer))

            # pipline for text generation
            lyrics_pipeline_top = pipeline('text-generation',model=model_top, tokenizer=tokenizer)
            song_input = text_prompt.get("1.0", 'end-1c') + '\n'

            result = lyrics_pipeline_top(song_input, min_length=int(entry_min.get()), max_length=int(entry_max.get()))[0]['generated_text']
            
            text_lyrics.delete('1.0', 'end') # delete last lyrics
            text_lyrics.insert("0.0", result) # insert generated lyrics
        case "both":
            # load model from file
            model_both = GPT2LMHeadModel.from_pretrained(savedModelBoth)
            model_both.resize_token_embeddings(len(tokenizer))

            # pipline for text generation
            lyrics_pipeline_both = pipeline('text-generation',model=model_both, tokenizer=tokenizer)
            song_input = text_prompt.get("1.0", 'end-1c') + '\n'

            result = lyrics_pipeline_both(song_input, min_length=int(entry_min.get()), max_length=int(entry_max.get()))[0]['generated_text']
            
            text_lyrics.delete('1.0', 'end') # delete last lyrics
            text_lyrics.insert("0.0", result) # insert generated lyrics
        case "gpt":
            ## load vanilla GPT2 from huggingface
            modelGPT2 = GPT2LMHeadModel.from_pretrained('gpt2')
            modelGPT2.resize_token_embeddings(len(tokenizer))

            # pipline for text generation
            gpt_pipeline = pipeline('text-generation',model=modelGPT2, tokenizer=tokenizer)
            song_input = text_prompt.get("1.0", 'end-1c') + '\n'

            result = gpt_pipeline(song_input, min_length=int(entry_min.get()), max_length=int(entry_max.get()))[0]['generated_text']

            text_lyrics.delete('1.0', 'end') # delete last lyrics
            text_lyrics.insert("0.0", result) # insert generated lyrics

# GUI Definition
label_model = customtkinter.CTkLabel(master=app, justify=tkinter.LEFT, text="Chose a model:", font=("Arial", 18))
label_model.grid(row=0, columnspan=4, pady=20)

radiobutton_var = tkinter.StringVar(value="rap")

radiobutton_rap = customtkinter.CTkRadioButton(master=app, variable=radiobutton_var, value="rap", text="Rap", font=("Arial", 14))
radiobutton_rap.grid(row=1, column=0, padx=(20), pady=5)

radiobutton_top = customtkinter.CTkRadioButton(master=app, variable=radiobutton_var, value="top", text="Top 100", font=("Arial", 14))
radiobutton_top.grid(row=1, column=1,pady=5)

radiobutton_both = customtkinter.CTkRadioButton(master=app, variable=radiobutton_var, value="both", text="Both", font=("Arial", 14))
radiobutton_both.grid(row=1, column=2,pady=5)

radiobutton_GPT_2 = customtkinter.CTkRadioButton(master=app, variable=radiobutton_var, value="gpt", text="GPT_2", font=("Arial", 14))
radiobutton_GPT_2.grid(row=1, column=3,padx=(20), pady=5)

label_prompt = customtkinter.CTkLabel(master=app, justify=tkinter.LEFT, text="Fill in a prompt:", font=("Arial", 18))
label_prompt.grid(row=2, columnspan=4, pady=20)

text_prompt = customtkinter.CTkTextbox(master=app, width=400, height=30)
text_prompt.grid(row=3, columnspan=4, padx=5)

label_length = customtkinter.CTkLabel(master=app, justify=tkinter.LEFT, text="Length of the generated lyrics:", font=("Arial", 18))
label_length.grid(row=4, columnspan=4, padx=5, pady=20)

entry_min = customtkinter.CTkEntry(master=app, placeholder_text="Min")
entry_min.grid(row=5, column=1, padx=5)

entry_max = customtkinter.CTkEntry(master=app, placeholder_text="Max")
entry_max.grid(row=5, column=2,  padx=5)

label_lyrics = customtkinter.CTkLabel(master=app, justify=tkinter.LEFT, text="The generated song lyrics:", font=("Arial", 18))
label_lyrics.grid(row=6, columnspan=4, padx=5, pady=20)

text_lyrics = customtkinter.CTkTextbox(master=app, width=450, height=200)
text_lyrics.grid(row=7, columnspan=4, padx=5, pady=5)

button_generate = customtkinter.CTkButton(master=app, command=lambda : threading.Thread(target=button_callback).start(), text="Generate song lyrics", font=("Arial", 18))
button_generate.grid(row=8, columnspan=4, pady=20)

app.mainloop()
