from collections import deque

test = deque(maxlen = 10)

for i in range(1,7):
    test.append(i)

print(test[-1]) if test else print("None")