import pandas as pd
import numpy as np
from scipy.optimize import linprog

# download the csv file
url = 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/c43f0d83d2b9d769c8dadb64959c16b01713cfaa/data/2024-25/players_raw.csv'
df = pd.read_csv(url, usecols=["first_name","second_name","points_per_game","total_points","element_type","now_cost"])

# Show the first 5 players on the list
#print(df.head(5))

# get the value and price of each player
v = df['total_points'].to_numpy()
p = df['now_cost'].to_numpy()

n = len(v)                                          # get the number of each player

o = np.ones(n,dtype=int)                            # create a vector of ones for number of players picked

positions = df['element_type'].to_numpy()           # get the positions of each player
p_ones_zeros = np.zeros((4, n), dtype=int)          # sort into a matrix of ones and zeros
p_ones_zeros[positions - 1, np.arange(n)] = 1       # row 0 -> gk, 1 -> def, 2 -> mid, 3 -> fwd
                                                    # each column represents a different player

c = -v                                              # cost is negative of the value (total points last season)
A_ub = p.reshape(1,n)                               # the price of each player, reshape to be a row vector
b_ub = 1000                                         # bounded by 100 million
A_eq = np.vstack([o, p_ones_zeros])                 # We need to pick 15 player, 2 gk, 5 def, 5 mid
b_eq = np.array([15,2,5,5,3])                       # 3 fwds. We do this with b_eq

# solve the equation with linprog
sol = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), integrality=1)


player_ids = np.where(sol.x == 1)                   # save the players ids

fnames = df['first_name']                           # get their names from the dataframe
snames = df['second_name']

gk = []
defend = []
mid = []
fwd = []

for i in player_ids[0]:                             # sort the players into their positions
    if(p_ones_zeros[0,i]==1):
        gk.append(i)
    elif(p_ones_zeros[1,i]==1):
        defend.append(i)
    elif(p_ones_zeros[2,i]==1):
        mid.append(i)
    elif(p_ones_zeros[3,i]==1):
        fwd.append(i)

# print the team
print("\nGoalkeepers:")
for i in gk:
    print(fnames[i],snames[i])
print("\nDefenders:")
for i in defend:
    print(fnames[i],snames[i])
print("\nMidfielders:")
for i in mid:
    print(fnames[i],snames[i])
print("\nStrikers:")
for i in fwd:
    print(fnames[i],snames[i])