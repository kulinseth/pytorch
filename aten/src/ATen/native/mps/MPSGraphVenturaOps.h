#pragma once
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

// TODO: Remove me when moved to MacOS 13
@interface MPSGraph (VenturaOps)

#if !defined(__MAC_13_0) && \
    (!defined(MAC_OS_X_VERSION_13_0) || (MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_13_0))

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

- (MPSGraphTensor * _Nonnull)cumulativeSumWithTensor:(MPSGraphTensor * _Nonnull)tensor
                                       axis:(NSInteger)axis
                                       name:(NSString * _Nullable)name;

- (MPSGraphTensor * _Nonnull)sortWithTensor:(MPSGraphTensor * _Nonnull)tensor
                                       axis:(NSInteger)axis
                                       name:(NSString * _Nullable)name;

- (MPSGraphTensor * _Nonnull)argSortWithTensor:(MPSGraphTensor * _Nonnull)tensor
                                       axis:(NSInteger)axis
                                       name:(NSString * _Nullable)name;

- (MPSGraphTensor * _Nonnull)inverseOfTensor:(MPSGraphTensor * _Nonnull) inputTensor
                                       name:(NSString * _Nullable)name;
@end
