// Fill out your copyright notice in the Description page of Project Settings.

#include "MyGameModeBase.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include "Kismet/GameplayStatics.h"

AMyGameModeBase::AMyGameModeBase()
{
	PrimaryActorTick.bCanEverTick = true;
}

void AMyGameModeBase::BeginPlay()
{
	Super::BeginPlay();
	PlayerController = GetWorld()->GetFirstPlayerController();
	
	// Setup TFActor pointer
	UGameplayStatics::GetAllActorsOfClass(GetWorld(), ATFAgent::StaticClass(), TFActors);
	TFActor = Cast<ATFAgent>(TFActors[0]);
}

void AMyGameModeBase::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	if (PlayerController->WasInputKeyJustPressed(EKeys::SpaceBar))
	{
		// Read the pixels from the RenderTarget and store them in a FColor array
		TArray<FColor> SurfData;
		FRenderTarget* RenderTarget = MyRenderTarget->GameThread_GetRenderTargetResource();
		RenderTarget->ReadPixels(SurfData);

		// Indexing
		FIntPoint XY = RenderTarget->GetSizeXY();
		// Saving test image and calculate gray scale
		std::ofstream ofile;
		ofile.open(TCHAR_TO_UTF8(*(FPaths::ProjectContentDir() + FString("RGB.txt"))), std::ios::trunc);

		for (FColor f : SurfData)
		{
			ofile << int(f.R) << "," << int(f.G) << "," << int(f.B) << std::endl;
		}
		ofile.close();

		// TF Inference
		// Prepare grayscale data
		TArray<float> img_data;
		TArray<float> out;
		for (FColor f : SurfData)
		{
			float x = (float(f.B) * 0.07 + float(f.G) * 0.72 + float(f.R) * 0.21) / 255.f;
			img_data.Add(x);
		}

		// Get prediction
		auto start = std::chrono::high_resolution_clock::now();
		TFActor->Inference(img_data, out);
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
		long long milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

		// Prepare string output
		if (GEngine) GEngine->AddOnScreenDebugMessage(-1, 3.0f, FColor::Yellow,
			FString::Printf(TEXT("Inference cost %d ms"), milliseconds), true, FVector2D(1.5, 1.5));

		FString result(" Cat        --        Dog\n");
		for (float f : out) {
			result += FString::Printf(TEXT("%.1f%%                "), f*100);
		}

		UE_LOG(LogTemp, Warning, TEXT("%s"), *result);
		if (GEngine) GEngine->AddOnScreenDebugMessage(-1, 3.0f, FColor::Yellow,
			result, true, FVector2D(1.5, 1.5));
	}
}
