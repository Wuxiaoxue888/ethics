from population import Population


def main():
    population = Population()
    for i in range(100):
        #population.print()
        print(f"-----------------Generation {i+1}-----------------------")
        population.order_by_fitness()
        #population.print()
        #print("---------------------------------------")
        population.create_offsprings(10, 10, selection_function="tournament", tournament_replacement=False)
        #population.print()
        #print("---------------------------------------")
        population.print()

if __name__ == '__main__':
    main()
