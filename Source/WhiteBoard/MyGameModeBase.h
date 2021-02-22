// Fill out your copyright notice in the Description page of Project Settings.

#pragma once
#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Engine/TextureRenderTarget.h"
#include "TFAgent.h"
#include "MyGameModeBase.generated.h"

/**
 * 
 */
UCLASS()
class WHITEBOARD_API AMyGameModeBase : public AGameModeBase
{
	GENERATED_BODY()

public:
	APlayerController* PlayerController;
	TArray<AActor*> TFActors;
	ATFAgent* TFActor;
	
	UPROPERTY(EditAnywhere)
	UTextureRenderTarget* MyRenderTarget;

	AMyGameModeBase();
	virtual void BeginPlay() override;
	virtual void Tick(float DeltaTime) override;
};
