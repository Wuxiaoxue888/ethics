#include "population.h"
#include <stdio.h>
#include <stdlib.h>

void initializePopulation(struct Population *p) {
  int i, j;
  for (i = 0; i < POPULATION_SIZE; i++) {
    p->Images[i] = (struct Image *)malloc(sizeof(struct Image));

    for (j = 0; j < CHROMOSOME_SIZE; j++) {
      p->Images[i]->chromosome[j] = rand() % (255 - 0 + 1) + 0;
    }
  }
}

void printPopulation(struct Population *p) {
  int i, j;
  for (i = 0; i < POPULATION_SIZE; i++) {
    printf("Image %d: (", i);
    for (j = 0; j < CHROMOSOME_SIZE; j++) {
      printf("%d,", p->Images[i]->chromosome[j]);
    }
    printf(")\n");
  }
}
