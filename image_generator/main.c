#include "population.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
  srand(time(NULL));

  struct Population p;
  initializePopulation(&p);
  printPopulation(&p);
}
