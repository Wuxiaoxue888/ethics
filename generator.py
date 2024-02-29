from population import Population


def main():
    population = Population(doping_size=50)
    for i in range(50):
        print(f"-----------------Generation {i + 1}-----------------------")
        population.create_offsprings(10, 10, selection_function="tournament", tournament_replacement=False)
        population.order_by_fitness()
        population.print()

    population.export_to_csv("result.csv")


if __name__ == '__main__':
    main()
