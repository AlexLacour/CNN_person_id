import random
import time

SIZE = 50
x = [random.random() for _ in range(SIZE)]
y1 = [0 for _ in range(SIZE)]
y2 = [1 for _ in range(SIZE)]

x_gen = (element for element in x)
y1_gen = (element for element in y1)
y2_gen = (element for element in y2)


def generator():
    while True:
        a = next(x_gen)
        b = next(y1_gen)
        c = next(y2_gen)
        yield a, [b, c]


gen = generator()
while True:
    print(next(gen))
    time.sleep(1)
    print("done")
