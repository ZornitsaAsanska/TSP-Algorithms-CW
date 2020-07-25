import random

def edgeNodesNonMetric(n):
    for i in range(n):
        for j in range((i+1),n):
            if i == 0 and j == 1:
                edge = int(random.random()*n) + 1
                file = open(f'{n}nodes', "w")
                file.write(f' {i} {j} {edge}' + '\n')
                file.close()
            else:
                edge = int(random.random()*n) + 1
                file = open(f'{n}nodes', "a")
                file.write(f' {i} {j} {edge}' + '\n')
                file.close()

def main():

    edgeNodesNonMetric(10)

if __name__ == '__main__':
    main()