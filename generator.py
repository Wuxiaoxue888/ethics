from population import Population


def main():
    for _ in range(3):
        population = Population()
        #population.print()
        #print("---------------------------------------")
        population.order_by_fitness()
        #population.print()
        #print("---------------------------------------")
        population.create_offsprings()
        #population.print()
        #print("---------------------------------------")


if __name__ == '__main__':
    main()
