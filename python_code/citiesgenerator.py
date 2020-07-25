import random

def XYnodes(n, Xlim, Ylim):
    for i in range(n):
        node = (random.random()*Xlim, random.random()*Ylim)
        if i == 0:
            file = open(f'cities{n}', "w")
            file.write(f' {node[0]} {node[1]}' + '\n')
            file.close()
        else:
            file = open(f'cities{n}', "a")
            file.write(f' {node[0]} {node[1]}' + '\n')
            file.close()

def main():

    XYnodes(10, 100, 100)

if __name__ == '__main__':
    main()
