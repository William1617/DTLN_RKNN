from rknn.api import RKNN
import soundfile as sf
import numpy as np

model1='./DTLN_1.onnx'
model2='./DTLN_2.onnx'
testwav='./testwav.wav'
RKNN_MODEL1 = 'newdtln_1.rknn'
RKNN_MODEL2 = 'newdtln_2.rknn'

in_audio,rate1=sf.read(testwav)

block_len=512
block_shift=128
block_num=(len(in_audio)-384)//block_shift
in_h1=np.zeros((1,128)).astype('float32')
in_c1=np.zeros((1,128)).astype('float32')
in_h2=np.zeros((1,128)).astype('float32')
in_c2=np.zeros((1,128)).astype('float32')

in_h3=np.zeros((1,128)).astype('float32')
in_c3=np.zeros((1,128)).astype('float32')
in_h4=np.zeros((1,128)).astype('float32')
in_c4=np.zeros((1,128)).astype('float32')
rknn_lite1=RKNN(verbose=True)
rknn_lite1.config(target_platform='rk3568')

rknn_lite2=RKNN(verbose=True)
rknn_lite2.config(target_platform='rk3568')

ret=rknn_lite1.load_onnx(model1,input_size_list=[[1,1,257],[1,128],[1,128],[1,128],[1,128]])
if ret !=0:
    exit(ret)

ret=rknn_lite2.load_onnx(model2,input_size_list=[[1,1,512],[1,128],[1,128],[1,128],[1,128]])
if ret !=0:
    exit(ret)

ret=rknn_lite1.build(do_quantization=False)
if ret != 0:
    print('build model1 failed!')
    exit(ret)

ret=rknn_lite2.build(do_quantization=False)
if ret != 0:
    print('build model2 failed!')
    exit(ret)

ret=rknn_lite1.export_rknn(RKNN_MODEL1)

ret=rknn_lite2.export_rknn(RKNN_MODEL2)

ret = rknn_lite1.init_runtime()
if ret != 0:
    print('Init runtime environment1 failed!')
    exit(ret)

ret = rknn_lite2.init_runtime()
if ret != 0:
    print('Init runtime environment2 failed!')
    exit(ret)
out_audio=np.zeros(len(in_audio))

for idx in range(block_num):
    in_buffer=in_audio[idx*block_shift:idx*block_shift+block_len]
    in_fft=np.fft.rfft(in_buffer)
    in_mag=np.abs(in_fft)
    in_phase=np.angle(in_fft)
    
    in_mag=np.reshape(in_mag,(1,1,-1)).astype('float32')

    outputs1=rknn_lite1.inference(inputs=[in_mag,in_h1,in_c1,in_h2,in_c2])
    in_h1=outputs1[1]
    
    in_c1=outputs1[2]
    in_h2=outputs1[3]
    in_c2=outputs1[4]
    out_block1=outputs1[0]

    estimated_complex=out_block1*np.exp(1j*in_phase)
    est_block=np.fft.irfft(estimated_complex).astype('float32')
    est_block=np.reshape(est_block,(1,1,-1)).astype('float32')

    outputs3=rknn_lite2.inference(inputs=[est_block,in_h3,in_c3,in_h4,in_c4])
    in_h3=outputs3[1]
    in_c3=outputs3[2]
    in_h4=outputs3[3]
    in_c4=outputs3[4]
    out_block3=outputs3[0]

    out_audio[idx*block_shift:idx*block_shift+block_len] +=np.squeeze(out_block3)
   # print(idx)

sf.write('testout.wav',out_audio,16000)
