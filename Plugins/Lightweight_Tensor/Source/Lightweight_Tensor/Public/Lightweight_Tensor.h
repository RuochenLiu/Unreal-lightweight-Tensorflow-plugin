// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

class LIGHTWEIGHT_TENSOR_API FLightweight_TensorModule : public IModuleInterface
{
public:
	void* DLLHandler;
	/** IModuleInterface implementation */
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;
};
