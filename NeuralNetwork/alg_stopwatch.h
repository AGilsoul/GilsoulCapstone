/******************************************************************************
 *  File: alg_stopwatch.h
 * 
 *  A header file of a utility class for measuring the running time of an 
 *  algorithm. Implemenation is adapted from that of the red Algorithms 4ed 
 *  textbook which is available at https://algs4.cs.princeton.edu/code/.
 * 
 *  Last modified by: Abdulmalek Al-Gahmi
 *  Last modified on: Jan 1, 2021
 ******************************************************************************/

#ifndef _ADV_ALG_STOP_WATCH_H_
#define _ADV_ALG_STOP_WATCH_H_

#include <chrono>

class StopWatch {
private:
  std::chrono::high_resolution_clock::time_point start;

public:
  StopWatch();
  double elapsed_time();
  void reset();
};

#endif
