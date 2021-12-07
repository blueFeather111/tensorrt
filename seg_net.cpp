#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <fstream>
#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include "logging.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <chrono>
#include <opencv2/opencv.hpp>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define BATCH_SIZE 1
#define CONF_THRESH 0.05

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 720;
static const int INPUT_W = 1280;
static const int OUTPUT_SIZE = 2* 720 * 1280;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "setNet";

using namespace nvinfer1;
using namespace std;

static Logger gLogger;

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    INetworkDefinition* network = builder->createNetworkV2(0U); //这里是不是size要确认一下

    // Create input tensor of shape { 3, INPUT_H, INPUT_W } with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W}); //输入size,这里是3通道
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../segNet.wts"); //wts文件对应的文件夹
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    //参数解释：data:input data, 64:output channel， DimsHW:kernel size
    //最后两个一个是kernel weights, 一个是biasWeights
    //ConvolutionNd表示多维卷积
    //参考https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_network_definition.html
    //conv1
    IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 8, DimsHW{5, 5}, weightMap["conv1.weight"], weightMap["conv1.bias"]); //map名称要改
    assert(conv1);
    conv1->setStrideNd(DimsHW{2, 2});
    conv1->setPaddingNd(DimsHW{2, 2});
    //bn1
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "bn1", 1e-5);
    //参数1:input tensor, 参数2: ActivationType
    //relu1
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    //conv2
    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), 16, DimsHW{5, 5}, weightMap["conv2.weight"], weightMap["conv2.bias"]); //map名称要改
    assert(conv2);
    conv2->setStrideNd(DimsHW{2, 2});
    conv2->setPaddingNd(DimsHW{2, 2});
    //bn2
    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), "bn2", 1e-5);

    //参数1:input tensor, 参数2: ActivationType
    //relu1
    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    //conv3
    IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2->getOutput(0), 32, DimsHW{5, 5}, weightMap["conv3.weight"], weightMap["conv3.bias"]); //map名称要改
    assert(conv3);
    conv3->setStrideNd(DimsHW{2, 2});
    conv3->setPaddingNd(DimsHW{2, 2});
    //bn1
    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), "bn3", 1e-5);

    //参数1:input tensor, 参数2: ActivationType
    //relu3
    IActivationLayer* relu3 = network->addActivation(*bn3->getOutput(0), ActivationType::kRELU);
    assert(relu3);

    //deconv1
    IDeconvolutionLayer* deconv1 = network->addDeconvolutionNd(*relu3->getOutput(0), 16, DimsHW{5,5}, weightMap["deconv1.weight"], weightMap["deconv1.bias"]);
    deconv1->setStrideNd(DimsHW{2, 2});
    deconv1->setPrePadding(DimsHW{2, 2});
    deconv1->setPostPadding(DimsHW{1, 1});
    assert(deconv1);
    //deconv1->setNbGroups(1);
    //bn4
    IScaleLayer* bn4 = addBatchNorm2d(network, weightMap, *deconv1->getOutput(0), "bn4", 1e-5);
    //relu4
    IActivationLayer* relu4 = network->addActivation(*bn4->getOutput(0), ActivationType::kRELU);

    //deconv2
    IDeconvolutionLayer* deconv2 = network->addDeconvolutionNd(*relu4->getOutput(0), 8, DimsHW{5,5}, weightMap["deconv2.weight"], weightMap["deconv2.bias"]);
    deconv2->setStrideNd(DimsHW{2, 2});
    deconv2->setPrePadding(DimsHW{2, 2});
    deconv2->setPostPadding(DimsHW{1, 1});
    //bn5
    IScaleLayer* bn5 = addBatchNorm2d(network, weightMap, *deconv2->getOutput(0), "bn5", 1e-5);
    //relu5
    IActivationLayer* relu5 = network->addActivation(*bn5->getOutput(0), ActivationType::kRELU);

    //deconv3
    IDeconvolutionLayer* deconv3 = network->addDeconvolutionNd(*relu5->getOutput(0), 2, DimsHW{5,5}, weightMap["deconv3.weight"], weightMap["deconv3.bias"]);
    deconv3->setStrideNd(DimsHW{2, 2});
    deconv3->setPrePadding(DimsHW{2, 2});
    deconv3->setPostPadding(DimsHW{1, 1});

    deconv3->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*deconv3->getOutput(0));
    cout << "finish markOutput" << endl; //debug


    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(128*(1 << 20)); //debug
    cout << "finish setting" << endl; //debug
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));

    auto start = std::chrono::system_clock::now();
    context.enqueue(batchSize, buffers, stream, nullptr);
    auto end = chrono::system_clock::now();
    cout <<"inference time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    //CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);


    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

//zero copy pipeline, 板子上CPU和GPU是共享内存的
void doInference_zeroCopy(IExecutionContext& context, float* input, float* output, int batchSize)
{
    //input: h_in, output: h_out
    //buffer[in] : d_in, buffer[out]: d_out
    cudaSetDeviceFlags(cudaDeviceMapHost);

    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    //void* buffers[2];
    float *d_in, *d_out;

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    //const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    //const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    //buffers[inputIndex] = d_in;
    //buffers[outputIndex] = d_out;

    //cout<<"inputIndex = " << inputIndex << endl;  //debug
    //cout<<"outputIndex = " << outputIndex << endl;  //debug

    //CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    //CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));


    // Allocates page-locked memory on the host.
    CHECK(cudaHostAlloc((void **)&input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaHostAllocMapped));
    CHECK(cudaHostAlloc((void **)&output, batchSize * OUTPUT_SIZE * sizeof(float), cudaHostAllocMapped));

    cudaHostGetDevicePointer((void**)&d_in, (void*)input, 0);
    cudaHostGetDevicePointer((void**)&d_out, (void*)output, 0);
//
//    //get device pointer from host memory. No allocation or memcpy
    //cudaHostGetDevicePointer((void**)&buffers[inputIndex], (void*)input, 0);
    //cudaHostGetDevicePointer((void**)&buffers[outputIndex], (void*)output, 0);
//    // Create stream
      cudaStream_t stream;
      CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    //CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));

    //auto start = std::chrono::system_clock::now();
    //context.enqueue(batchSize, buffers, stream, nullptr);
    //auto end = chrono::system_clock::now();
    //cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    //CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));

    //cudaStreamSynchronize(stream);


    // Release stream and buffers
    cudaStreamDestroy(stream);
    //CHECK(cudaFree(buffers[inputIndex]));
    //CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv)
{
    char *trtModelStream{nullptr};
    size_t size{0};
    string engine_name = "segNet.engine";

    if (argc == 2 && std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p(engine_name, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    } else if (argc == 3 && std::string(argv[1]) == "-d") {
        ifstream file(engine_name, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        cerr << "arguments not right!" << endl;
        cerr << "./seg_net -s  //serialize model to engine" << endl;
        cerr << "./seg_net -d ../samples  //run inference" << endl;
        return -1;
    }

    vector<string> file_names;
    if (read_files_in_dir(argv[2], file_names) < 0) {
        cout << "read_files_in_dir failed. " << endl;
        return -1;
    }

    //input data
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    int fcount = 0;
    for(int f = 0; f < (int)file_names.size(); f++) {
        fcount ++;
        if(fcount < BATCH_SIZE && f+1 != (int)file_names.size()) continue; //fcount到batch_size, 一次取Batch size个图像
        for(int b = 0; b < fcount; b++){  //batch_size
            cout << file_names[f-fcount+1+b] << endl;
            cv::Mat img = cv::imread(string(argv[2]) + "/" + file_names[f-fcount+1+b]);
            if(img.empty()) continue;
            cv::Mat pr_img = img;  //fcn时预处理只做resize,seg_net不做
            int i = 0;
            for(int row = 0; row < INPUT_H; ++row) {
                uchar* uc_pixel = pr_img.data + row * pr_img.step; //img.step:一行的字节数，图像是按行储存的
                for(int col = 0; col < INPUT_W; ++col) {
                    data[b*3*INPUT_H*INPUT_W + i] = (float)(uc_pixel[2] / 255.0 - 0.5) * 2.0; //R
                    data[b*3*INPUT_H*INPUT_W + i + INPUT_H*INPUT_W] = (float)(uc_pixel[1] / 255.0 - 0.5) * 2.0; //G
                    data[b*3*INPUT_H*INPUT_W + i + 2*INPUT_H*INPUT_W] = (float)(uc_pixel[0] / 255.0 - 0.5) * 2.0; //B
                    uc_pixel += 3;
                    ++ i;
                }
            }
        }
        //run inference
        auto start = std::chrono::system_clock::now();
        //doInference_zeroCopy(*context, data, prob, BATCH_SIZE); //zero copy
        doInference(*context, data, prob, BATCH_SIZE);
        auto end = chrono::system_clock::now();
        cout << "total time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;

        float numerator = 0.0;

        numerator = exp(prob[0]);

        cv::Mat mask_mat = cv::Mat(INPUT_H, INPUT_W, CV_8UC1);

        //cv::Mat norm_mat = cv::Mat(INPUT_H, INPUT_W, CV_8UC1);

        uchar* ptmp = NULL;
        for(int i = 0; i < INPUT_H; i++) {
            ptmp = mask_mat.ptr<uchar>(i);
            for(int j = 0; j < INPUT_W; j++) {
                float numerator = exp(prob[i*INPUT_W + j]);
                float pixel = numerator / (numerator + exp(prob[i*INPUT_W + j + INPUT_W*INPUT_H])); //softmax
                if(pixel > CONF_THRESH) {
                    ptmp[j] = 255;
                } else {
                    ptmp[j] = 0;
                }
            }
        }

        //cv::normalize(mask_mat, norm_mat, 0, 255, cv::NORM_MINMAX, -1, cv::noArray());
        cv::imwrite("s_" + file_names[f-fcount+1] + "_segNet.jpg", mask_mat);

        fcount = 0;

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}
