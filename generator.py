from population import Population


def main():
    population = Population()
    for i in range(3):
        #population.print()
        print(f"-----------------Generation {i+1}-----------------------")
        population.order_by_fitness()
        #population.print()
        #print("---------------------------------------")
        population.create_offsprings()
        #population.print()
        #print("---------------------------------------")
        population.print()

if __name__ == '__main__':
    main()
