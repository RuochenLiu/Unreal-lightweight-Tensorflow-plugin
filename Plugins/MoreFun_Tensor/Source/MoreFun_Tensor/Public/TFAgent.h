#pragma once
#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "GameFramework/Actor.h"
#include "TensorModel.h"
#include "Tensor.h"
#include "c_api.h"

#include "TFAgent.generated.h"

UCLASS(hideCategories = (Rendering, Actor, LOD, Cooking, Transform, Input, Collision, Replication), BlueprintType)
class MOREFUN_TENSOR_API ATFAgent : public AActor
{
	GENERATED_BODY()

public:
	ATFAgent();

	// Model PB file name in {ProjectDir/Content/TensorFlow/}
	UPROPERTY(Category = "Default", BlueprintReadWrite, EditAnywhere)
	FString ModelName = "XXX";

	// Weight CKPT file name in {ProjectDir/Content/TensorFlow/}
	// UPROPERTY(Category = "Default", BlueprintReadWrite, EditAnywhere)
	// FString WeightsName = "XXX";

	// Whether load pre-trained weights
	UPROPERTY(Category = "Default", BlueprintReadWrite, EditAnywhere)
	bool LoadWeights = 0;

	// Input tensor name in graph
	UPROPERTY(Category = "Variable", BlueprintReadWrite, EditAnywhere)
	FString InputTensorName;

	// Output tensor name in graph
	UPROPERTY(Category = "Variable", BlueprintReadWrite, EditAnywhere)
	FString OutputTensorName;

	// Function to get output from pre-trained model
	UFUNCTION(BlueprintCallable)
	void Inference(TArray<float> new_data, TArray<float>& out);

	TensorModel* SelfModel;

	virtual void BeginPlay() override;
	virtual void Tick(float DeltaTime) override;
};
