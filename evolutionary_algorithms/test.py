def hamming_distance(string1, string2):
    distance = 0
    for i in range(len(string1)):
        if string1[i] != string2[i]:
            distance += 1
    return distance


target = '1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111'
sample = '1111111110111111110110011110000101110011111011101111111101111010101000111011101110100011111111100111'

hamming = hamming_distance(target, sample)
# hamming_near = scipy.spatial.distance.hamming(target, near)
fitnes = int(len(target) - hamming)
print('Hamming', hamming)
print('fitnes', fitnes)
print()

#plt.plot(range(len(self.best_fitness_hypotheses)), self.best_fitness_hypotheses)
#plt.show()