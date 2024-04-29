
 
#include"DTLNImpl.h"

#include "DTLN_defs.h"
#include "pocketfft_hdronly.h"
#include "AudioFile.h"

using namespace std;

typedef complex<double> cpx_type;

void ExportWAV(
        const std::string & Filename, 
		const std::vector<float>& Data, 
		unsigned SampleRate) {
    AudioFile<float>::AudioBuffer Buffer;
	Buffer.resize(1);

	Buffer[0] = Data;
	size_t BufSz = Data.size();

	AudioFile<float> File;
	File.setAudioBuffer(Buffer);
	File.setAudioBufferSize(1, (int)BufSz);
	File.setNumSamplesPerChannel((int)BufSz);
	File.setNumChannels(1);
	File.setBitDepth(16);
	File.setSampleRate(SAMEPLERATE);
	File.save(Filename, AudioFileFormat::Wave);		
}

void DTLN_RKNN(){
	

	trg_engine* m_pEngine;

    m_pEngine = new trg_engine;

    m_pEngine->statesa.clear();
    m_pEngine->statesb.clear();

    m_pEngine->statesa.resize(4);
    m_pEngine->statesb.resize(4);
    for (int i=0;i<4;i++){
        m_pEngine->statesa[i].clear();
        m_pEngine->statesb[i].clear();
        m_pEngine->statesa[i].resize(DTLN_UNIT_SIZE);
        m_pEngine->statesb[i].resize(DTLN_UNIT_SIZE);
        memset(m_pEngine->statesa[i].data(),0,DTLN_UNIT_SIZE*sizeof(float));
        memset(m_pEngine->statesb[i].data(),0,DTLN_UNIT_SIZE*sizeof(float));

    }

    int ret=0;
    int model_len1 = 0;
    unsigned char* model1 =  __ReadModel(DTLNModelNameA, &model_len1);
    rknn_context  m_ctx[2];
	rknn_input_output_num                       m_ioNum[2];
    ret=rknn_init(&(m_ctx[0]),model1,model_len1,0,NULL);
    if (ret < 0) {
        printf("rknn_init fail1! ret=%d\n", ret);
        return -1;
    }
    
    int model_len2 = 0;
    unsigned char* model2 =  __ReadModel(DTLNModelNameB, &model_len2);

    ret=rknn_init(&(m_ctx[1]),model2,model_len2,0,NULL);
    if (ret < 0) {
        printf("rknn_init fail2! ret=%d\n", ret);
        return -1;
    }

    ret = rknn_query(m_ctx[0], RKNN_QUERY_IN_OUT_NUM, &(m_ioNum[0]), sizeof(rknn_input_output_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query1 fail1! ret=%d\n", ret);
        return -1;
    }

    ret = rknn_query(m_ctx[1], RKNN_QUERY_IN_OUT_NUM, &(m_ioNum[1]), sizeof(rknn_input_output_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query1 fail2! ret=%d\n", ret);
        return -1;
    }
   
    rknn_tensor_attr input_attrs_a[m_ioNum[0].n_input];  
    memset(input_attrs_a,0,m_ioNum[0].n_input*sizeof(rknn_tensor_attr));
   
    rknn_tensor_attr input_attrs_b[m_ioNum[1].n_input];
    memset(input_attrs_b,0,m_ioNum[1].n_input*sizeof(rknn_tensor_attr));

    int i=0;
    for (i=0;i<m_ioNum[0].n_input;i++){
       
        input_attrs_a[i].index=i;
        ret= rknn_query(m_ctx[0], RKNN_QUERY_INPUT_ATTR, &(input_attrs_a[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_init error1! ret=%d\n", ret);
            return -1;
        }
        
        input_attrs_b[i].index=i;
        ret= rknn_query(m_ctx[1], RKNN_QUERY_INPUT_ATTR, &(input_attrs_b[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_init error3! ret=%d\n", ret);
            return -1;
        }   
    }
    rknn_tensor_attr output_attrs_a[m_ioNum[0].n_output], output_attrs_b[m_ioNum[1].n_output];
    memset(output_attrs_a,0,m_ioNum[0].n_output*sizeof(rknn_tensor_attr));
    memset(output_attrs_b,0,m_ioNum[1].n_output*sizeof(rknn_tensor_attr));

    for (i=0;i<m_ioNum[0].n_output;i++){
        output_attrs_a[i].index=i;
        ret = rknn_query(m_ctx[0], RKNN_QUERY_OUTPUT_ATTR, &(output_attrs_a[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query2 fail1! ret=%d\n", ret);
            return -1;
        }   
        output_attrs_b[i].index=i;
        ret = rknn_query(m_ctx[1], RKNN_QUERY_OUTPUT_ATTR, &(output_attrs_b[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query2 fail3! ret=%d\n", ret);
            return -1;
        }
    }

	float f32_output[DTLN_BLOCK_LEN];
    std::vector<float>  testoutdata; //vector used to store enhanced data in a wav file
    AudioFile<float> inputwavfile;
    std::string testfile="./wav/testwav.wav";
    inputwavfile.load(testfile);
    int audiolen=inputfile.getNumSamplesPerChannel();
    int process_num=(audiolen-DTLN_BLOCK_LEN)/DTLN_BLOCK_SHIFT;

    for(int i=0;i<process_num;i++)
    {
        memmove(m_pEngine->in_buffer, m_pEngine->in_buffer + DTLN_BLOCK_SHIFT, (DTLN_BLOCK_LEN - DTLN_BLOCK_SHIFT) * sizeof(float));
      
        for(int n=0;n<BLOCK_SHIFT;n++){
                m_pEngine->in_buffer[n+DTLN_BLOCK_LEN-DTLN_BLOCK_SHIFT]=inputmicfile.samples[0][n+i*DTLN_BLOCK_SHIFT];
            } 
        RKNNInfer(m_pEngine,m_ctx,m_ioNum);
        for(int j=0;j<DTLN_BLOCK_SHIFT;j++){
            testoutdata.push_back(m_pEngine->out_buffer[j]);    //for one forward process save first BLOCK_SHIFT model output samples
        }
    }
    ExportWAV("rknntest.wav",testoutdata,SAMEPLERATE);
for (int i=0;i<2:i++){
        if(m_ctx[i]>0){
            rknn_destroy(m_ctx[i]);
        }
    }
    if (model1) {
        free(model1);
    }
    if (model2) {
        free(model2);
    }
    delete m_pEngine;
	
}

 
void RKNNInfer(trg_engine* m_pEngine, rknn_context* m_ctx,rknn_input_output_num *m_ionum) {

	float in_mag[FFT_OUT_SIZE] ={0}; ;
    float in_phase[FFT_OUT_SIZE] = { 0 };
    float estimated_block[DTLN_BLOCK_LEN] ={0};

    double fft_in[DTLN_BLOCK_LEN];
    std::vector<cpx_type> fft_res(DTLN_BLOCK_LEN);

	std::vector<size_t> shape;
    shape.push_back(DTLN_BLOCK_LEN);
    std::vector<size_t> axes;
    axes.push_back(0);
    std::vector<ptrdiff_t> stridel, strideo;
    strideo.push_back(sizeof(cpx_type));
    stridel.push_back(sizeof(double));
    float out_mask[FFT_OUT_SIZE] ={0};
    float out_block[DTLN_BLOCK_LEN]={0} ;
 
    int res2;

	for (int i = 0; i < DTLN_BLOCK_LEN; i++){
        fft_in[i] = m_pEngine->in_buffer[i];
	}

	pocketfft::r2c(shape, stridel, strideo, axes, pocketfft::FORWARD, fft_in, fft_res.data(), 1.0);

	__calc_mag_phase(fft_res, in_mag, in_phase, FFT_OUT_SIZE);

    rknn_input   input_details_a[m_ioNum[0].n_input], input_details_b[m_ioNum[1].n_input];
    rknn_output  output_details_a[m_ioNum[0].n_output], output_details_b[m_ioNum[1].n_output];
    memset(input_details_a,0,m_ioNum[0].n_input*sizeof(rknn_input));
    memset(input_details_b,0,m_ioNum[1].n_input*sizeof(rknn_input));

    for (int i=0;i<m_ioNum[0].n_input;i++){
        input_details_a[i].index=i;
       
        input_details_a[i].type=RKNN_TENSOR_FLOAT32;
        input_details_a[i].fmt=RKNN_TENSOR_NHWC;

        input_details_b[i].index=i;
        
        input_details_b[i].type=RKNN_TENSOR_FLOAT32;
        input_details_b[i].fmt=RKNN_TENSOR_NHWC;

        if(i<1){
            input_details_a[0].size=FFT_OUT_SIZE*sizeof(float);
            input_details_b[0].size=DTLN_BLOCK_LEN*sizeof(float);
        }else{
            input_details_a[i].size=DTLN_UNIT_SIZE*sizeof(float);
            input_details_b[i].size=DTLN_UNIT_SIZE*sizeof(float);
        }
    }
    memset(output_details_a,0,m_ioNum[0].n_output*sizeof(rknn_output));
    memset(output_details_b,0,m_ioNum[1].n_output*sizeof(rknn_output));
  //  memset(output_details_b,0,m_ioNum[1].n_output*sizeof(rknn_output));
    
    for (int i=0;i<m_ioNum[0].n_output;i++){
        output_details_a[i].index=i;
        output_details_a[i].want_float=1;
        output_details_a[i].is_prealloc = 0;
        
        output_details_b[i].index=i;
        output_details_b[i].want_float=1;
        output_details_b[i].is_prealloc = 0;
    }

    input_details_a[0].buf=in_mag;
   
    for (int i=0;i<4;i++){
        m_input_details_a[1+i].buf=m_pEngine->statesa[i].data();

    }
   
    res2=rknn_inputs_set(m_ctx[0], m_ioNum[0].n_input,input_details_a);

   if (res2 < 0) {
        LOGW("%s:Model a can't set input,break",__FUNCTION__);
        ResetInout();
        return -1;
    }
    
    res2=rknn_run(m_ctx[0], nullptr);
    
    res2 = rknn_outputs_get(m_ctx[0], m_ioNum[0].n_output,output_details_a, NULL);

    out_mask =(float*)(output_details_a[0].buf);
    for (int i=0;i<4;i++){
        memcpy(m_pEngine->statesa[i].data(),(float*)(m_output_details_a[i+1].buf),DTLN_UNIT_SIZE*sizeof(float));
    }
    //memcpy(m_pEngine->states_a,outstatea,DTLN_STATE_SIZE*sizeof(float));

	for (int i = 0; i < FFT_OUT_SIZE; i++) {
        fft_res[i] = cpx_type( out_mask[i] * cosf(in_phase[i]), out_mask[i] * sinf(in_phase[i]));
	}

    pocketfft::c2r(shape, strideo, stridel,axes, pocketfft::BACKWARD, fft_res.data(), fft_in, 1.0);

    for (int i = 0; i < DTLN_BLOCK_LEN; i++)
    {
        estimated_block[i] = fft_in[i] / DTLN_BLOCK_LEN;
    }
    res2 = rknn_outputs_release(m_ctx[0], m_ioNum[0].n_output, output_details_a);
    
    input_details_b[0].buf = estimated_block;
    for (int i=0;i<4;i++){
        m_input_details_b[1+i].buf=m_pEngine->statesb[i].data();

    }

    res2=rknn_inputs_set(m_ctx[1], m_ioNum[1].n_input, input_details_b);

    res2=rknn_run(m_ctx[1], nullptr);
  
    res2 = rknn_outputs_get(m_ctx[1], m_ioNum[1].n_output, output_details_b, NULL);
  //  res2 =rknn_query(m_ctx[1], RKNN_QUERY_PERF_RUN, &(runtime2), sizeof(rknn_perf_run));
   // m_duration +=runtime2.run_duration;


    out_block =(float*)(output_details_b[0].buf);
    
    for (int i=0;i<4;i++){
        memcpy(m_pEngine->statesb[i].data(),(float*)(m_output_details_b[i+1].buf),DTLN_UNIT_SIZE*sizeof(float));
    }
    
    res2 = rknn_outputs_release(m_ctx[1], m_ioNum[1].n_output, output_details_b);
    
    memmove(m_pEngine->out_buffer, 
        m_pEngine->out_buffer + DTLN_BLOCK_SHIFT, 
        (DTLN_BLOCK_LEN - DTLN_BLOCK_SHIFT) * sizeof(float));
    memset(m_pEngine->out_buffer + (DTLN_BLOCK_LEN - DTLN_BLOCK_SHIFT), 0, DTLN_BLOCK_SHIFT * sizeof(float));
    for (int i = 0; i < DTLN_BLOCK_LEN; i++){
        m_pEngine->out_buffer[i] += out_block[i];
    }
   
}
 
void __calc_mag_phase(std::vector<cpx_type> fft_res, 
				float* in_mag, float* in_phase, int count) {
    for (int i = 0; i < count; i++) {
        in_mag[i] = sqrtf(fft_res[i].real() * fft_res[i].real() + fft_res[i].imag() * fft_res[i].imag());
        in_phase[i] = atan2f(fft_res[i].imag(), fft_res[i].real());
    }
}

unsigned char* __ReadModel(char *filename, int *model_size) {
    FILE *fp = fopen(filename, "rb");
    if(fp == nullptr) {
        printf("fopen %s fail!\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    unsigned char *model = (unsigned char*)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if(model_len != fread(model, 1, model_len, fp)) {
        printf("fread %s fail!\n", filename);
        free(model);
        return NULL;
    }
    
    *model_size = model_len;
    if(fp) {
        fclose(fp);
    }

    return model;
}



