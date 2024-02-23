#ifndef POPULATION
#define POPULATION

#define POPULATION_SIZE 1
#define CHROMOSOME_SIZE 784

struct Image {
  int chromosome[CHROMOSOME_SIZE];
  float fitness;
};

struct Population {
  struct Image *Images[POPULATION_SIZE];
};

void initializePopulation(struct Population *p);
void printPopulation(struct Population *p);
void freePopulation(struct Population *p);

#endif
