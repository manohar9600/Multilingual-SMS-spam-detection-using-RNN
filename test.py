import pandas as pd

data = open('Resources/temp.txt','r')
data = data.read()
data = [sen for sen in data.split('\n')]

file = open('Resources/abbr.text', 'w')

j = 0

while j < len(data):
    print(str(j))
    while data[j] == '':
        j = j + 1
    file.write(data[j])
    j = j + 1
    file.write(',')
    while data[j] == '':
        j = j + 1
    file.write(data[j])
    j = j + 1
    file.write('\n')

file.close()
