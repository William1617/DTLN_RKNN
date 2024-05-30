
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
#define MODEL_INPUT_NUM (5)
#define MODEL_OUTPUT_NUM (5)


struct trg_engine {
    float in_buffer[DTLN_BLOCK_LEN] = { 0 };
    float out_buffer[DTLN_BLOCK_LEN] = { 0 };
  
    std::vector<std::vector<float>> statesa;
    std::vector<std::vector<float>> statesb;
    
   
};
struct trg_engine_zerocopy {
    float in_buffer[DTLN_BLOCK_LEN] = { 0 };
    float out_buffer[DTLN_BLOCK_LEN] = { 0 };
  
    std::vector<std::vector<float>> statesa;
    std::vector<std::vector<float>> statesb;
    
    rknn_tensor_mem*  m_input_details_a[MODEL_INPUT_NUM];
    rknn_tensor_mem*  m_input_details_b[MODEL_INPUT_NUM];
    rknn_tensor_mem*  m_output_details_a[MODEL_OUTPUT_NUM];
    rknn_tensor_mem*  m_output_details_b[MODEL_OUTPUT_NUM];

};


