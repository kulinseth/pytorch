#pragma once
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

// TODO: Remove me when moved to MacOS 13
@interface MPSGraph (VenturaOps)

#if !defined(__MAC_13_0)
API_AVAILABLE(macos(13.0))
typedef NS_ENUM(NSUInteger, MPSGraphResizeNearestRoundingMode)
{
    MPSGraphResizeNearestRoundingModeRoundPreferCeil   =  0L,
    MPSGraphResizeNearestRoundingModeRoundPreferFloor  =  1L,
    MPSGraphResizeNearestRoundingModeCeil              =  2L,
    MPSGraphResizeNearestRoundingModeFloor             =  3L,
    MPSGraphResizeNearestRoundingModeRoundToEven       =  4L,
    MPSGraphResizeNearestRoundingModeRoundToOdd        =  5L,
};
#endif

- (MPSGraphTensor *)cumulativeSumWithTensor:(MPSGraphTensor *)tensor
                                       axis:(NSInteger)axis
                                       name:(NSString *)name;

- (MPSGraphTensor *)sortWithTensor:(MPSGraphTensor *)tensor
                                       axis:(NSInteger)axis
                                       name:(NSString *)name;

- (MPSGraphTensor *)argSortWithTensor:(MPSGraphTensor *)tensor
                                       axis:(NSInteger)axis
                                       name:(NSString *)name;
@end
