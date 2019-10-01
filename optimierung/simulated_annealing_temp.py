import math
import random

# Mit abnehmender Temperatur sinkt die Wahrscheinlichkeit für Bergabschritte

for i in [1000, 100, 10, 5, 1]:
    exp = math.exp((-32 - (-28)) / i)
    print('Temperature:', i, 'exp value:', exp)
    bergabschritte = [random.random() < exp for j in range(100)]
    print('Bergabschritte:', bergabschritte.count(True))
