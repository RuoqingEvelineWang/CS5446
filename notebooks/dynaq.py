#reward function: SOFA score measures patient health (higher score/increase in score should be penalised)
#referenced from: Deep Reinforcement Learning for Sepsis Treatment
def calculate_reward(cur_SOFA, next_SOFA, cur_MAP, next_MAP):
    reward = 0
    c0 = -0.025
    c1 = -0.125
    c2 = 0.4
    map_reward = 0
    if cur_SOFA == next_SOFA and next_SOFA > 0:
        reward += c0
    if cur_MAP == 3 and next_MAP == 3: #MAP stays at normal level
        map_reward = 0
    elif cur_MAP == next_MAP: #MAP stays at low level
        map_reward = (-3 + cur_MAP) * 0.5
    elif cur_MAP > next_MAP: #MAP deteriorates
        map_reward = (next_MAP - cur_MAP) * (4 - cur_MAP) * 0.8
    elif cur_MAP < next_MAP: #MAP improves
        map_reward = next_MAP - cur_MAP
    return reward + c1 * (next_SOFA - cur_SOFA) + c2 * map_reward

#calculate SOFA score for each state (using binned values)
def calculate_sofa_score(MAP, urine, ALT, AST, lactic_acid, serum_creatinine, p_f_ratio, GCS_total):
    cardiovascular_score = 0
    if MAP < 3:
        cardiovascular_score = 1
    
    respitory_score = 0
    if p_f_ratio == 0:
        respitory_score = 4
    elif p_f_ratio == 1:
        respitory_score = 3
    elif p_f_ratio == 2:
        respitory_score = 2
    else:
        respitory_score = 1
    
    renal_score = serum_creatinine

    #divide daily urine output standards by 6 as the data is 4 hour interval
    if urine == 1:
        renal_score = max(renal_score, 3)
    if urine == 0:
        renal_score = max(renal_score, 4)

    #since bilirubin is not available, use ALT and AST to calculate liver score
    liver_score = 0
    if (ALT > 0 and AST > 0):
        liver_score = 1
    if (ALT > 1 or AST > 1):
        liver_score = 2
    if (ALT > 2 or AST > 2):
        liver_score = 3

    neuro_score = 0
    if GCS_total == 0:
        neuro_score = 4
    elif GCS_total == 1:
        neuro_score = 3
    elif GCS_total  == 2:
        neuro_score = 2
    elif GCS_total == 3:
        neuro_score = 1
    
    lactic_acid_score = 0
    if lactic_acid > 0:
        lactic_acid_score = 2
    
    return cardiovascular_score + respitory_score + renal_score + liver_score + neuro_score + lactic_acid_score

import pandas as pd
import numpy as np

state_columns = ['MAP', 'diastolic_bp', 'systolic_bp', 'urine', 'ALT', 'AST', 'lactic_acid', 'serum_creatinine', 'p_f_ratio', 'GCS_total']
action_columns = ['fluid_boluses', 'vasopressors']

def create_transitions(df):
    X = []
    y = []
    patients = df['PatientID'].unique()

    for patient in patients:
        patient_records = df[df['PatientID'] == patient].reset_index(drop=True)
        for i in range(len(patient_records) - 1):
            current_state = patient_records.iloc[i][state_columns]
            next_state = patient_records.iloc[i + 1][state_columns]
            action = patient_records.iloc[i][action_columns]
            X.append(np.concatenate([current_state, action]))
            y.append(next_state)
    return pd.DataFrame(X, columns=state_columns + action_columns), pd.DataFrame(y, columns=state_columns)

df = pd.read_csv('binned_df.csv')

X, y = create_transitions(df)
print(X.head())
print(y.head())

#use K nearest neighbours to calculate next state prediction (function approximation for transitions)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn_models = {col: KNeighborsClassifier(n_neighbors=20) for col in state_columns}

for col in state_columns:
    knn_models[col].fit(X_train, y_train[col])

#evaluate accuracy
predictions = pd.DataFrame()
for col in state_columns:
    predictions[col] = knn_models[col].predict(X_test)
    accuracy = accuracy_score(y_test[col], predictions[col])
    print(f"Accuracy for predicting {col}: {accuracy:.2f}")

unique_states = df[state_columns].drop_duplicates().sort_values(by=state_columns).reset_index(drop=True)
state_dict = {tuple(row): idx for idx, row in enumerate(unique_states.values)}
unique_actions = df[action_columns].drop_duplicates().sort_values(by=action_columns).reset_index(drop=True)
action_dict = {tuple(row): idx for idx, row in enumerate(unique_actions.values)}
inverse_action_dict = {v: k for k, v in action_dict.items()}
inverse_state_dict = {v: k for k, v in state_dict.items()}


def get_state_index(row):
    return state_dict[tuple(row[state_columns])]

def get_action_index(row):
    return action_dict[tuple(row[action_columns])]

def get_action_from_index(index):
    return inverse_action_dict[index]

def find_closest_index(input_tuple):
    closest_index = None
    min_distance = float('inf')
    
    for value, index in state_dict.items():
        distance = np.linalg.norm(np.array(value) - np.array(input_tuple))
        
        if distance < min_distance:
            min_distance = distance
            closest_index = index
    
    return closest_index

seed = 42
np.random.seed(seed)
train_ratio = 0.8
num_patients = df['PatientID'].nunique()
num_train_samples = int(num_patients * train_ratio)
train_ids = np.random.choice(num_patients, num_train_samples, replace=False)
train_df = df[df["PatientID"].isin(train_ids)].reset_index(drop=True)
test_df = df[~df["PatientID"].isin(train_ids)].reset_index(drop=True)
print(f'Train Data: {train_df.shape}')
print(f'Test Data: {test_df.shape}')

def calc_sofa_score_from_row(row):
    return calculate_sofa_score(row['MAP'].item(), row['urine'].item(), row['ALT'].item(), row['AST'].item(), row['lactic_acid'].item(), row['serum_creatinine'].item(), row['p_f_ratio'].item(), row['GCS_total'].item())

def row_to_state_action_df(row):
    state_action = {
        'MAP': row['MAP'].item(),
        'diastolic_bp': row['diastolic_bp'].item(),
        'systolic_bp': row['systolic_bp'].item(),
        'urine': row['urine'].item(),
        'ALT': row['ALT'].item(),
        'AST': row['AST'].item(),
        'lactic_acid': row['lactic_acid'].item(),
        'serum_creatinine': row['serum_creatinine'].item(),
        'p_f_ratio':row['p_f_ratio'].item(),
        'GCS_total': row['GCS_total'].item(),
        'fluid_boluses': row['fluid_boluses'].item(),
        'vasopressors': row['vasopressors'].item()
    }
    return pd.DataFrame([state_action])

def list_to_state_action_df(l):
    state_action = {
        'MAP': l[0],
        'diastolic_bp': l[1],
        'systolic_bp': l[2],
        'urine': l[3],
        'ALT': l[4],
        'AST': l[5],
        'lactic_acid': l[6],
        'serum_creatinine': l[7],
        'p_f_ratio': l[8],
        'GCS_total': l[9],
        'fluid_boluses': l[10],
        'vasopressors': l[11]
    }
    return pd.DataFrame([state_action])

import random

alpha = 0.1
gamma = 0.99
#number of simulated experiences per real experience
num_simulations = 2
num_iterations = 5

q_table = np.zeros((len(state_dict), len(action_dict)))

def dyna_q_train(train_df, num_iterations, num_simulations):
    
    global q_table
    for episode in range(num_iterations):
        print(f"Episode {episode + 1}/{num_iterations}")
        patients = train_df['PatientID'].unique()
        i = 0
        for patient in patients:
            print(f"Patient {i}")
            i+=1
            patient_data = train_df[train_df['PatientID'] == patient].reset_index(drop=True)
            
            for t in range(len(patient_data) - 1):
                
                current_row = patient_data.iloc[t]
                next_row = patient_data.iloc[t + 1]
                current_state_index = get_state_index(current_row)
                action_index = get_action_index(current_row)
                
                predicted_next_state = pd.DataFrame()
                for col in state_columns:
                    predicted_next_state[col] = knn_models[col].predict(row_to_state_action_df(current_row))

                reward = calculate_reward(
                    calc_sofa_score_from_row(current_row),
                    calc_sofa_score_from_row(predicted_next_state),
                    current_row['MAP'].item(),
                    predicted_next_state['MAP'].item()
                )
                
                next_state_index = get_state_index(next_row)

                best_next_q = np.max(q_table[next_state_index])
                q_table[current_state_index, action_index] += alpha * (
                    reward + gamma * best_next_q - q_table[current_state_index, action_index])
                for _ in range(num_simulations):
                    sim_state_index = random.choice(list(state_dict.values()))
                    sim_action_index = random.choice(list(action_dict.values()))

                    sim_state = pd.Series(inverse_state_dict[sim_state_index], index=state_columns)
                    sim_action = pd.Series(get_action_from_index(sim_action_index), index=action_columns)
                    sim_next_state = pd.DataFrame()
                    for col in state_columns:
                        sim_next_state[col] = knn_models[col].predict(list_to_state_action_df(sim_state.tolist() + sim_action.tolist()))
                    sim_reward = calculate_reward(
                        calc_sofa_score_from_row(sim_state),
                        calc_sofa_score_from_row(sim_next_state),
                        sim_state['MAP'].item(),
                        sim_next_state['MAP'].item()
                    )

                    if tuple(sim_next_state.iloc[0]) in state_dict:
                        sim_next_state_index = state_dict[tuple(sim_next_state.iloc[0])]
                    else:
                        sim_next_state_index = find_closest_index(tuple(sim_next_state.iloc[0]))

                    best_sim_next_q = np.max(q_table[sim_next_state_index])
                    q_table[sim_state_index, sim_action_index] += alpha * (
                        sim_reward + gamma * best_sim_next_q - q_table[sim_state_index, sim_action_index])

    return q_table

q_table = dyna_q_train(train_df, num_iterations=1, num_simulations=5)
print("Training completed.")
q_table_df = pd.DataFrame(q_table)
q_table_df.to_csv("q_table.csv")
print("Saved to csv.")