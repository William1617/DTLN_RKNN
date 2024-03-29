
#include <cstdio>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <assert.h>
#include <stdint.h>
#include <string.h>


#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <dlfcn.h>

#include "rknn_api.h"


#define DTLN_MODEL_A "dtln_1.rknn"
#define DTLN_MODEL_B "dtln_2.rknn"

#define SAMEPLERATE  (16000)
#define DTLN_BLOCK_LEN		(512)
#define DTLN_BLOCK_SHIFT  (128)
#define DTLN_STATE_SIZE  (512)
#define DTLN_UNIT_SIZE (128)
#define FFT_OUT_SIZE    (DTLN_BLOCK_LEN / 2 + 1)


struct trg_engine {
    float in_buffer[DTLN_BLOCK_LEN] = { 0 };
    float out_buffer[DTLN_BLOCK_LEN] = { 0 };
  
    std::vector<std::vector<float>> statesa;
    std::vector<std::vector<float>> statesb;
    
   
};

