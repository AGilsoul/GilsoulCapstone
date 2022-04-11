/******************************************************************************
 *  File: alg_stopwatch.cpp
 * 
 *  An implementation of a utility class for measuring the running time of an 
 *  algorithm. Implemenation is adapted from that of the red Algorithms 4ed 
 *  textbook which is available at https://algs4.cs.princeton.edu/code/.
 * 
 *  Last modified by: Abdulmalek Al-Gahmi
 *  Last modified on: Jan 1, 2021
 ******************************************************************************/

#include "../include/alg_stopwatch.h"

StopWatch::StopWatch(): start(std::chrono::high_resolution_clock::now()) {}

double StopWatch::elapsed_time(){
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> running_time = end - start;
  return running_time.count();
}

void StopWatch::reset(){
  start = std::chrono::high_resolution_clock::now();
}
