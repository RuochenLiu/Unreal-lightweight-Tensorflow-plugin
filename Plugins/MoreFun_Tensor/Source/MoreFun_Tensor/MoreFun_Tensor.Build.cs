// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;

public class MoreFun_Tensor : ModuleRules
{
	public MoreFun_Tensor(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicIncludePaths.AddRange(
			new string[] {
				ModuleDirectory + "/Public",
				ModuleDirectory + "/ThirdParty/libtensorflow/include/tensorflow/c",
			}
			);
				
		
		PrivateIncludePaths.AddRange(
			new string[] {
				ModuleDirectory + "/Private",
				ModuleDirectory + "/ThirdParty/libtensorflow/include/tensorflow/c",
			}
			);


		PublicDependencyModuleNames.AddRange(
		   new string[]
		   {
				"Core",
				"InputCore",
		   }
		   );


		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				"CoreUObject",
				"Engine",
			}
			);

		PublicDelayLoadDLLs.Add(PluginDirectory + "/Binaries/Win64/tensorflow.dll");
		PublicAdditionalLibraries.Add(PluginDirectory + "/Binaries/Win64/tensorflow.lib");
	}
}
