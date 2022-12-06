import tkinter
import customtkinter

customtkinter.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

app = customtkinter.CTk()
app.geometry("800x700")
app.title("Song Lyrics Generator")

def button_callback():
#     print("Button click", combobox_1.get())
    print("Button click")

frame_1 = customtkinter.CTkFrame(master=app)
frame_1.pack(pady=20, padx=40, fill="both", expand=True)

label_1 = customtkinter.CTkLabel(master=frame_1, justify=tkinter.LEFT, text="Chose a model:")
label_1.pack(pady=10, padx=10)

radiobutton_var = tkinter.IntVar(value=1)

radiobutton_1 = customtkinter.CTkRadioButton(master=frame_1, variable=radiobutton_var, value=1, text="Rap")
radiobutton_1.pack(pady=10, padx=10)

radiobutton_2 = customtkinter.CTkRadioButton(master=frame_1, variable=radiobutton_var, value=2, text="Top 100")
radiobutton_2.pack(pady=10, padx=10)

radiobutton_3 = customtkinter.CTkRadioButton(master=frame_1, variable=radiobutton_var, value=3, text="Both")
radiobutton_3.pack(pady=10, padx=10)

radiobutton_4 = customtkinter.CTkRadioButton(master=frame_1, variable=radiobutton_var, value=3, text="GPT_2")
radiobutton_4.pack(pady=10, padx=10)

label_2 = customtkinter.CTkLabel(master=frame_1, justify=tkinter.LEFT, text="Fill in a prompt:")
label_2.pack(pady=10, padx=10)

text_1 = customtkinter.CTkTextbox(master=frame_1, width=400, height=30)
text_1.pack(pady=10, padx=10)
text_1.insert("0.0", "40 Jahre die Flippers")

label_3 = customtkinter.CTkLabel(master=frame_1, justify=tkinter.LEFT, text="Length of the generated lyrics:")
label_3.pack(pady=10, padx=10)

entry_1 = customtkinter.CTkEntry(master=frame_1, placeholder_text="Min")
entry_1.pack(pady=10, padx=10)

entry_2 = customtkinter.CTkEntry(master=frame_1, placeholder_text="Max")
entry_2.pack(pady=10, padx=10)

label_4 = customtkinter.CTkLabel(master=frame_1, justify=tkinter.LEFT, text="The generated song lyrics:")
label_4.pack(pady=10, padx=10)

text_2 = customtkinter.CTkTextbox(master=frame_1, width=400, height=100)
text_2.pack(pady=10, padx=10)
text_2.insert("0.0", "Blubb\n\n\n\n")

button_1 = customtkinter.CTkButton(master=frame_1, command=button_callback, text="Generate song lyrics")
button_1.pack(pady=10, padx=10)

app.mainloop()