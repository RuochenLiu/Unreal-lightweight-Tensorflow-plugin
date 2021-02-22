#include "TFAgent.h"

ATFAgent::ATFAgent()
{
    PrimaryActorTick.bCanEverTick = true;
}

void ATFAgent::BeginPlay()
{
    Super::BeginPlay();

    // TensorFlow API version check
    UE_LOG(LogTemp, Warning, TEXT("Powered by TensorFlow C library version %s\n"), UTF8_TO_TCHAR(TF_Version()));

    FString ModelPath = FPaths::ProjectContentDir() + "TensorFlow/" + ModelName + ".pb";
    // FString CheckPointPath = FPaths::ProjectContentDir() + "TensorFlow/" + WeightsName + ".ckpt";

    // Load tf model and checkpoint weights
    SelfModel = TensorModel::Instance(TCHAR_TO_UTF8(*ModelPath));
    // LoadWeights ? SelfModel->restore(TCHAR_TO_UTF8(*CheckPointPath)) : SelfModel->init();
}

void ATFAgent::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);
}

void ATFAgent::Inference(TArray<float> new_data, TArray<float> &out)
{
    // Setup simple input and output
    Tensor input{ *SelfModel, TCHAR_TO_UTF8(*InputTensorName) };
    Tensor output{ *SelfModel, TCHAR_TO_UTF8(*OutputTensorName) };

    // Fill input with raw data (Auto-resize by input tensor shape)
    std::vector<float> raw;
    for (float f : new_data)
    {
        raw.push_back(f);
    }
    input.set_data(raw);

    // Inference
    SelfModel->run({ &input }, output);

    for (float f : output.get_data<float>())
    {
        out.Add(f);
    }
}