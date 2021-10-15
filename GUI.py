from tkinter.font import NORMAL
from chatPyMe import predict_answer, answer, intents
import tkinter as tk
from tkinter import DISABLED, END

def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete('0.0', END)
    
    if msg != ' ':
        # User design part
        Chatbox.config(state=NORMAL)
        Chatbox.insert(END, "You: " + msg + '\n\n')
        Chatbox.config(foreground="#421EE4", font=('Verdana', 12))
        
        # Bot process part
        intent = predict_answer(msg)
        response = answer(intent=intent, intent_json=intents)
        
        Chatbox.insert(END, 'Xavier: ' + response + '\n\n')
        
        Chatbox.config(state=DISABLED)
        Chatbox.yview(END)

window = tk.Tk()
window.title("Hi – I'm Xavier")
window.geometry("400x500")
window.resizable(width=False, height=False)

# Create chat window
Chatbox = tk.Text(window, bd=0, bg="white", height='8', width="50", font='Poppings')
Chatbox.config(state=DISABLED)

# Bind scrollbar to chat window
scrollbar = tk.Scrollbar(window, command=Chatbox.yview, cursor='heart')
Chatbox['yscrollcommand'] = scrollbar.set

# Create the box to enter message
EntryBox = tk.Text(window, bd=0, bg='#ECECEC', width="29", height="5", font='Georgia')
# EntryBox.bind('<Return>', send)

# Create send Button
SendButton = tk.Button(window, font=("Verdana", 12, 'bold'), text="Chat", width="12", height="5",
                                        bd=1, background='#0AF028', activebackground="#1976CC", activeforeground='#FFFFFF' ,
                                        fg="#000000" ,command= send)

# Place all the components on the screen
Chatbox.place(x=6, y=6, height=386, width=370)
scrollbar.place(x=376, y=6, height=386)
EntryBox.place(x=6, y=401, height= 50, width=300)
SendButton.place(x=311, y=401, height= 50, width=80)

window.mainloop()
