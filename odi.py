import streamlit as st
import pandas as pd
import pickle 

st.title('ODI Match Predictor System')

odi_dict = pickle.load(open('odi_dict1.pkl' , 'rb'))
odi_dict1 = pd.DataFrame(odi_dict)

options = st.selectbox(' TEAM 1 ', odi_dict1['Team 1'].unique())

options1 = st.selectbox(' TEAM 2', odi_dict1['Team 2'].unique())
df = pd.read_csv(r'C:\Users\jayes\Downloads\ODI-data-1971-2017.csv')
df = df.drop(['Scorecard','Match Date'],axis=1)
df = df.dropna()

won_by_runs = []

won_by_wickets = []

for margin in df['Margin'].astype(str):
    splitted_data = margin.split(' ')
    try:
        index = splitted_data.index('runs')
        
        won_by_runs.append(eval(splitted_data[0]))
        
        won_by_wickets.append(0)
    except:
        print('-')
        
        
    try:
        index = splitted_data.index('run')
        
        won_by_runs.append(eval(splitted_data[0]))
        
        won_by_wickets.append(0)
    except:
        print('-')
        
    try:
        index = splitted_data.index('wickets')
        
        won_by_wickets.append(eval(splitted_data[0]))
        
        won_by_runs.append(0)
    except:
        print('-')
        
        
    try:
        index = splitted_data.index('wicket')
        
        won_by_wickets.append(eval(splitted_data[0]))
        
        won_by_runs.append(0)
    except:
        print('-')
        
        
df['won_by_runs'] = won_by_runs


df['won_by_wickets'] = won_by_wickets



winning_team = []

for team1 , team2 , winner in zip(df['Team 1'] , df['Team 2'] , df['Winner']):
    
    if winner==team1:
        winning_team.append(team1)
    
    
    if winner==team2:
        winning_team.append(team2)
        
df['winning_team'] = winning_team
           

team1_first_batting = []


team2_first_batting = []

for  team1 , team2 , winner , runs,wickets  in zip(df['Team 1'],df['Team 2'] , df['Winner'] ,df['won_by_runs'],df['won_by_wickets']):
  
    if  runs >0 and winner==team1:
        

        team1_first_batting.append(1)
        
        team2_first_batting.append(0)
        
   
    if runs > 0 and winner==team2:
        team2_first_batting.append(1)
        
        team1_first_batting.append(0)
  
    
   
    if wickets > 0 and winner==team1:
        
        team1_first_batting.append(0)
        
        team2_first_batting.append(1)
        

    
    if wickets> 0 and winner==team2:
        
        team1_first_batting.append(1)
        
        team2_first_batting.append(0)
        
df['Team_1_first_batting'] = team1_first_batting

df['Team_2_first_batting'] = team2_first_batting

winning_team = []


for  team1 , team2 , winner  in zip(df['Team 1'],df['Team 2'] , df['Winner']):
    
    if winner==team1:
        
        winning_team.append(1)
        
        

        
    if winner==team2:
        
        winning_team.append(2)

        

df['Winning_team'] = winning_team

from sklearn.preprocessing import MinMaxScaler
def scale_data(df,col):
    # Instantiate MinMaxScaler
    mx = MinMaxScaler()
    
    df[col] =mx.fit_transform(df[col])
    
    return df
sc_data = ['won_by_runs','won_by_wickets']

# passing data and name for scaling
scale_data(df,sc_data)

data = pd.get_dummies(df,columns=['Team 1','Team 2'])
data = data.copy()

x = data.drop(['Winner','Margin','Ground','Winning_team','winning_team'],axis=1)

y = data['Winning_team']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2)

from sklearn.metrics import  accuracy_score


from sklearn.svm import SVC
svc_classifier = SVC()
svc_classifier.fit(x_train, y_train)

sc51 = svc_classifier.score(x_train, y_train)

sc52 = svc_classifier.score(x_test, y_test)

print('Acc of training set  = ',sc51)

print('Acc of test set  = ',sc52)

predicted_winner = svc_classifier.predict(x)[0]




if st.button('Predict the winner'):
        st.write(predicted_winner)