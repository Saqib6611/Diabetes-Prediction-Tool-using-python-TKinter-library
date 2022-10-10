# Important Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
plt.style.use('ggplot')
print('load libraries-done')
# Upload The dataset
set=pd.read_csv("D:\project\diabetes.csv")
data=set.copy()
print(f'There are {data.shape[0]} rows and {data.shape[1]} columns')
# CLEAN THE DATASET
#handling missing values
data.loc[data.Glucose==0,'Glucose']=data.Glucose.mean()
data.loc[data.BloodPressure==0,'BloodPressure']=data.BloodPressure.mean()
data.loc[data.SkinThickness==0,'SkinThickness']=data.SkinThickness.mean()
data.loc[data.Insulin==0,'Insulin']=data.Insulin.median()
data.loc[data.DiabetesPedigreeFunction==0,'DiabetesPedigreeFunction']=data.DiabetesPedigreeFunction.median()
data.loc[data.BMI==0,'BMI']=data.BMI.mean()
# Resampling the dataset
zero  = data[data['Outcome']==0]   #zero values in outcome column
one = data[data['Outcome']==1]  # one values in outcome column
from sklearn.utils import resample
#minority class that 1, we need to upsample/increase that class so that there is no bias
#n_samples = 500 means we want 500 sample of class 1, since there are 500 samples of class 0
data_minority_upsampled = resample(one, replace = True, n_samples = 500) 
#concatenate
data = pd.concat([zero, data_minority_upsampled])
# Shuffle after resampling
from sklearn.utils import shuffle
data = shuffle(data)
# Train Test Split
x=data.drop(columns='Outcome')
y=data['Outcome']
#x=data.iloc[:,:-1].values
#y=data.iloc[:,7].values
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=0)
print(x_train.shape, x_test.shape)
# Training Model using Random Forest ensembler
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=10)
model.fit(x_train, y_train) #fitting the model 
y_pred = model.predict(x_test) #prediction
y_train_pred = model.predict(x_train)
# Printing Performaces results
print("classifiaction report:\n",classification_report(y_test, y_pred))
print("confusion_matrix:\n",confusion_matrix(y_test, y_pred))
# INTERFACE MAKING
#importing important libraries

import tkinter as tk
from tkinter import *
import csv
from tkinter import scrolledtext
from tkinter import messagebox as mb

# Declearing text entries

fields ='Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DPF','Age',

#function for adding new data in new dataset

def Addnew(entries):
    import array as arr   
    text1=[]
    for entry in entries:
        field = entry[0]
        text  = entry[1].get()
        text2=(text)
        text1.append(text2)
    input=[text1]
    output=model.predict(input)
    text1.append(output[0])
    with open('new_diabetes1.csv','a',newline='') as csvFile:
        write = csv.writer(csvFile)
        write.writerow(text1)
        csvFile.close()
        
#function for reading values from new dataset 

def Newentries():
    root = tk.Tk()
    root.title("New Entries")
    s = tk.Scrollbar(root)
    T = tk.Text(root)
    
    T.focus_set()
    s.pack(side=tk.RIGHT, fill=tk.Y)
    T.pack(side=tk.LEFT, fill=tk.Y)
    s.config(command=T.yview)
    T.config(bg='#021c1e',foreground='white',yscrollcommand=s.set)
    i=0
    j=0
    with open('new_diabetes1.csv','r') as File:
            reader = csv.reader(File)
            for row in reader:
                i=i+1
                for items in row:
                    
                    if i== 1: #display dataset header
                        j=j+1
                        items = chr(64+j) +" :- " + items 
                        T.insert('end', items )
                        T.insert('end','\n')
                    else:
                        st = "%6s" %items
                        T.insert('end', st)
                        T.insert('end',' ')
                        
                T.insert('end','\n')
                if i==1:
                    for k in range(j):
                        colname = chr(65+k)
                        st = "%6s" %colname
                        T.insert('end', st)
                        T.insert('end',' ')
                    T.insert('end','\n')
                    T.insert('end','________________________________________________________________')
                    T.insert('end','\n\n')

def fetch(entries):
    import array as arr
    
    text1=[]
    
    for entry in entries:
        field = entry[0]
        text  = entry[1].get()
        text2=(text)
        text1.append(text2)
    input=[text1]
    output=model.predict(input)
    print(output)
    if output==1:
        predict()
    else:
        nopredict()
        
#declearing function for  showing messages

def predict():
    mb.showinfo("Caution", "You Are Requested That Please Consult a Specialist as Your Diagnostic Reports Are Pointing Towards The Health Issue.")
def nopredict():
    mb.showinfo("Negative", "disease not predicted, no need to worry")
def callback():
    if mb.askyesno('Verify', 'Are you sure, You want to quit?'):
        command=root.destroy()
    else:
        mb.showinfo('No', 'Quit has been cancelled')

#declearing function for reading the contents of doctor file

def Doctors():
        
    window = tk.Tk()
    window.title("Doctor's List")
    window.geometry('422x500')
    
    txt = scrolledtext.ScrolledText(window,foreground='white',bg='#004445',width=50,height=30)
    txt.grid(column=0,row=0)
    txt.insert(INSERT,"List of consultants:\n_________________________________________\n\n")
    i=0
    with open('C:/Users\Saqib Gulzar Bhat\Desktop\doctor.txt', 'r') as f:
    
        for line in f:
            if (i==0):
                txt.insert(INSERT, "Name : ")
                txt.insert(INSERT,line)
            elif (i==1):
                txt.insert(INSERT, "Address : ")
                txt.insert(INSERT,line)
            elif (i==2):
                txt.insert(INSERT, "Contact : ")   
                txt.insert(INSERT,line)
            elif (i==3):
                txt.insert(INSERT, "Timing : ")
                txt.insert(INSERT,line)
            
            if (i == 3):
                i = 0
                txt.insert(INSERT, "\n\n")
            else:
                i=i+1
                
#declearing function for displaying the dataset on which our model was built.

def disp():
    root = tk.Tk()
    root.title("Historical Data")
    s = tk.Scrollbar(root)
    T = tk.Text(root)
    
    T.focus_set()
    s.pack(side=tk.RIGHT, fill=tk.Y)
    T.pack(side=tk.LEFT, fill=tk.Y)
    s.config(command=T.yview)
    T.config(bg='#021c1e',foreground='white',yscrollcommand=s.set)
    i=0
    j=0
    with open('C:/Users\Saqib Gulzar Bhat\Desktop\diabetes.csv') as File:
            reader = csv.reader(File)
            for row in reader:
                i=i+1
                for items in row:
                    
                    if i== 1: #display dataset header
                        j=j+1
                        items = chr(64+j) +" :- " + items 
                        T.insert('end', items )
                        T.insert('end','\n')
                    else:
                        st = "%6s" %items
                        T.insert('end', st)
                        T.insert('end',' ')
                        
                T.insert('end','\n')
                if i==1:
                    for k in range(j):
                        colname = chr(65+k)
                        st = "%6s" %colname
                        T.insert('end', st)
                        T.insert('end',' ')
                    T.insert('end','\n')
                    T.insert('end','________________________________________________________________')
                    T.insert('end','\n\n')
    root.mainloop()
    
#declearing a function for making textbox widgets.
        
def makeform(root, fields):
    entries = []
    for field in fields:
        row = tk.Frame(root)
        lab = tk.Label(row, width=15,bg="#3b4d61",foreground='white', text=field, anchor='w')
        ent = tk.Entry(row)
        row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        lab.pack(side=tk.LEFT)
        ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
        entries.append((field, ent))
    return entries

#main function of the program

if __name__ == '__main__':
    root = tk.Tk()
    root.title('Diapredict')
    textt = tk.Text(root,bg='#1868ae', height=4, width=158)
    textt.insert(tk.END,'\n                                                   Diapredict- A Model That Predicts Diabetes \n', 'big')
    textt.tag_configure('bold_italics', font=('Arial', 20, 'bold', 'italic'))
    textt.tag_configure('big',foreground='white', font=('Verdana', 15, 'bold'))
    textt.pack(side=tk.TOP)
    
    
    text1 = tk.Text(root, height=29, width=50)
    
    #inserting image in our interface
    
    photo = tk.PhotoImage(file='C:/Users/Saqib Gulzar Bhat/Desktop/monophy.gif')
    text1.insert(tk.END, '\n')
    text1.image_create(tk.END, image=photo)

    text1.pack(side=tk.LEFT)
    
    
     #inserting text in our interface
        
    text2 = tk.Text(root, height=29, width=50)
    scroll = tk.Scrollbar(root, command=text2.yview)
    text2.configure(yscrollcommand=scroll.set)
    text2.tag_configure('bold_italics', font=('Arial', 20, 'bold', 'italic'))
    text2.tag_configure('big', font=('Verdana', 20, 'bold'))
    text2.tag_configure('color',
                        foreground='#1868ae',
                        font=('Tempus Sans ITC', 13, 'bold'))
    text2.tag_bind('follow',
               '<1>',
               lambda e, t=text2: t.insert(tk.END, "Not now, maybe later!"))
    text2.insert(tk.END,'\n  LARRY KING\n', 'big')
    quote = """
    Diabetes just boggles me.I know 
    when you get a heart pain; I've
    had them. I don't know what diabetes
    feels like. ....if someone had
    said to me, "What's your number 1 health
    problem?" I would have said heart
    disease and then diabetes.And 
    what doctors tell me now is that
    I can transpose them and say
    diabetes first.
    """
    text2.insert(tk.END, quote, 'color')
    #text2.insert(tk.END, 'follow-up\n', 'follow')
    text2.pack(side=tk.LEFT)
    scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    #making Buttons for our interface
    
    textt1 = tk.Text(root, bg='#021c1e', height=6, width=54)
    textt1.insert(tk.END,'''   Pregnencies(0-17)      Glucose(56-199)      Blood Pressure(22-124)
    
   Skin Thickenss(7-99)   Insulin(14-846)       BMI(18-67.1)
   
   DPF(0.07-2.4)              Age(21-81)
   
   Please enter the values in range given above''', 'big')
    #textt1.insert(tk.END,'\nFill The Boxex with With Numeric Values Only And make a prediction \n', 'big')
    textt1.tag_configure('bold_italics', font=('Arial', 8))
    textt1.tag_configure('big', foreground='white', font=('Verdana', 8))
    textt1.pack(side=tk.TOP)
    
    ents = makeform(root, fields)
    root.bind('<Return>', (lambda event, e=ents: fetch(e)))   
    b1 = tk.Button(root, text='Predict',bg='#77c593',foreground='white',
                  command=(lambda e=ents: fetch(e)))
    b1.pack(side=tk.LEFT, padx=5, pady=5)
    b3 = tk.Button(text='Doctors',bg='#77c593',foreground='white', command=Doctors)
    b3.pack(side=tk.LEFT, padx=5, pady=5)
    b4 = tk.Button(text='Historical data',bg='#77c593',foreground='white', command=disp)
    b4.pack(side=tk.LEFT, padx=5, pady=5)
    b5 = tk.Button(text='Add Data',bg='#77c593',foreground='white', command=(lambda e=ents: Addnew(e)))
    b5.pack(side=tk.LEFT, padx=5, pady=5)
    b6 = tk.Button(text='show new data',bg='#77c593',foreground='white', command=Newentries)
    b6.pack(side=tk.LEFT, padx=5, pady=5)
    b2 = tk.Button(text='Quit',bg='#77c593',foreground='white', command=callback)
    b2.pack(side=tk.LEFT, padx=5, pady=5)
    root.mainloop()
