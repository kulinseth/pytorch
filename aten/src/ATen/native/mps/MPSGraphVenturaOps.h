#pragma once
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#if !defined(__MAC_13_0) && !defined(MAC_OS_X_VERSION_13_0)
typedef NSUInteger MPSGraphResizeNearestRoundingMode_Ventura NS_TYPED_ENUM;
MPSGraphResizeNearestRoundingMode_Ventura const MPSGraphResizeNearestRoundingModeRoundPreferCeil = 0L;
MPSGraphResizeNearestRoundingMode_Ventura const MPSGraphResizeNearestRoundingModeRoundPreferFloor = 1L;
MPSGraphResizeNearestRoundingMode_Ventura const MPSGraphResizeNearestRoundingModeCeil = 2L;
MPSGraphResizeNearestRoundingMode_Ventura const MPSGraphResizeNearestRoundingModeFloor = 3L;
#elif !defined(__MAC_13_2) && !defined(MAC_OS_X_VERSION_13_2)
typedef NSUInteger MPSGraphResizeNearestRoundingMode_Ventura NS_TYPED_ENUM;
MPSGraphResizeNearestRoundingMode_Ventura const MPSGraphResizeNearestRoundingModeRoundToEven = 4L;
MPSGraphResizeNearestRoundingMode_Ventura const MPSGraphResizeNearestRoundingModeRoundToOdd = 5L;
#elif
typedef MPSGraphResizeNearestRoundingMode MPSGraphResizeNearestRoundingMode_Ventura;
#endif

// TODO: Remove me when moved to MacOS 13
@interface MPSGraph (VenturaOps)

- (MPSGraphTensor * _Nonnull)cumulativeSumWithTensor:(MPSGraphTensor * _Nonnull)tensor
                                                axis:(NSInteger)axis
                                                name:(NSString * _Nullable)name;

- (MPSGraphTensor * _Nonnull)sortWithTensor:(MPSGraphTensor * _Nonnull)tensor
                                       axis:(NSInteger)axis
                                       name:(NSString * _Nullable)name;

- (MPSGraphTensor * _Nonnull) sortWithTensor:(MPSGraphTensor * _Nonnull) tensor
                               axis:(NSInteger) axis
                         descending:(BOOL) descending
                               name:(NSString * _Nullable) name;

- (MPSGraphTensor * _Nonnull) sortWithTensor:(MPSGraphTensor * _Nonnull) tensor
                         axisTensor:(MPSGraphTensor * _Nonnull) axisTensor
                         descending:(BOOL) descending
                               name:(NSString * _Nullable) name;

- (MPSGraphTensor * _Nonnull) sortWithTensor:(MPSGraphTensor * _Nonnull) tensor
                         axisTensor:(MPSGraphTensor * _Nonnull) axisTensor
                               name:(NSString * _Nullable) name;

- (MPSGraphTensor * _Nonnull)argSortWithTensor:(MPSGraphTensor * _Nonnull)tensor
                                          axis:(NSInteger)axis
                                          name:(NSString * _Nullable)name;

- (MPSGraphTensor * _Nonnull) argSortWithTensor:(MPSGraphTensor * _Nonnull) tensor
                                  axis:(NSInteger) axis
                            descending:(BOOL) descending
                                  name:(NSString * _Nullable) name;

- (MPSGraphTensor * _Nonnull) argSortWithTensor:(MPSGraphTensor * _Nonnull) tensor
                           axisTensor:(MPSGraphTensor * _Nonnull) axisTensor
                           descending:(BOOL) descending
                                 name:(NSString * _Nullable) name;

- (MPSGraphTensor * _Nonnull) argSortWithTensor:(MPSGraphTensor * _Nonnull) tensor
                           axisTensor:(MPSGraphTensor * _Nonnull) axisTensor
                                 name:(NSString * _Nullable) name;

- (MPSGraphTensor * _Nonnull)inverseOfTensor:(MPSGraphTensor * _Nonnull) inputTensor
                                        name:(NSString * _Nullable)name;

- (MPSGraphTensor * _Nonnull) resizeNearestWithTensor:(MPSGraphTensor * _Nonnull) imagesTensor
                                           sizeTensor:(MPSGraphTensor * _Nonnull) size
                                  nearestRoundingMode:(MPSGraphResizeNearestRoundingMode) nearestRoundingMode
                                         centerResult:(BOOL) centerResult
                                         alignCorners:(BOOL) alignCorners
                                               layout:(MPSGraphTensorNamedDataLayout) layout
                                                 name:(NSString * _Nullable) name;

- (MPSGraphTensor * _Nonnull) resizeNearestWithTensor:(MPSGraphTensor * _Nonnull) imagesTensor
                                           sizeTensor:(MPSGraphTensor * _Nonnull) size
                                    scaleOffsetTensor:(MPSGraphTensor * _Nonnull) scaleOffset
                                  nearestRoundingMode:(MPSGraphResizeNearestRoundingMode) nearestRoundingMode
                                               layout:(MPSGraphTensorNamedDataLayout) layout
                                                 name:(NSString * _Nullable) name;

- (MPSGraphTensor * _Nonnull) resizeBilinearWithTensor:(MPSGraphTensor * _Nonnull) imagesTensor
                                            sizeTensor:(MPSGraphTensor * _Nonnull) size
                                          centerResult:(BOOL) centerResult
                                          alignCorners:(BOOL) alignCorners
                                                layout:(MPSGraphTensorNamedDataLayout) layout
                                                  name:(NSString * _Nullable) name;

- (MPSGraphTensor * _Nonnull) resizeBilinearWithTensor:(MPSGraphTensor * _Nonnull) imagesTensor
                                            sizeTensor:(MPSGraphTensor * _Nonnull) size
                                     scaleOffsetTensor:(MPSGraphTensor * _Nonnull) scaleOffset
                                                layout:(MPSGraphTensorNamedDataLayout) layout
                                                  name:(NSString * _Nullable) name;

- (MPSGraphTensor * _Nonnull) resizeNearestWithGradientTensor:(MPSGraphTensor * _Nonnull) gradient
                                                        input:(MPSGraphTensor * _Nonnull) input
                                          nearestRoundingMode:(MPSGraphResizeNearestRoundingMode) nearestRoundingMode
                                                 centerResult:(BOOL) centerResult
                                                 alignCorners:(BOOL) alignCorners
                                                       layout:(MPSGraphTensorNamedDataLayout) layout
                                                         name:(NSString * _Nullable) name;

- (MPSGraphTensor * _Nonnull) resizeNearestWithGradientTensor:(MPSGraphTensor * _Nonnull) gradient
                                                        input:(MPSGraphTensor * _Nonnull) input
                                            scaleOffsetTensor:(MPSGraphTensor * _Nonnull) scaleOffset
                                          nearestRoundingMode:(MPSGraphResizeNearestRoundingMode) nearestRoundingMode
                                                       layout:(MPSGraphTensorNamedDataLayout) layout
                                                         name:(NSString * _Nullable) name;

- (MPSGraphTensor * _Nonnull) resizeBilinearWithGradientTensor:(MPSGraphTensor * _Nonnull) gradient
                                                         input:(MPSGraphTensor * _Nonnull) input
                                                  centerResult:(BOOL) centerResult
                                                  alignCorners:(BOOL) alignCorners
                                                        layout:(MPSGraphTensorNamedDataLayout) layout
                                                          name:(NSString * _Nullable) name;

- (MPSGraphTensor * _Nonnull) resizeBilinearWithGradientTensor:(MPSGraphTensor * _Nonnull) gradient
                                                         input:(MPSGraphTensor * _Nonnull) input
                                             scaleOffsetTensor:(MPSGraphTensor * _Nonnull) scaleOffset
                                                        layout:(MPSGraphTensorNamedDataLayout) layout
                                                          name:(NSString * _Nullable) name;

- (MPSGraphTensor * _Nonnull) sampleGridWithSourceTensor:(MPSGraphTensor * _Nonnull) source
                                        coordinateTensor:(MPSGraphTensor * _Nonnull) coordinates
                                                  layout:(MPSGraphTensorNamedDataLayout) layout
                                    normalizeCoordinates:(BOOL) normalizeCoordinates
                                     relativeCoordinates:(BOOL) relativeCoordinates
                                            alignCorners:(BOOL) alignCorners
                                             paddingMode:(MPSGraphPaddingMode) paddingMode
                                            samplingMode:(MPSGraphResizeMode) samplingMode
                                           constantValue:(double) constantValue
                                                    name:(NSString * _Nullable) name;

- (MPSGraphTensor * _Nonnull) sampleGridWithSourceTensor:(MPSGraphTensor * _Nonnull) source
                                        coordinateTensor:(MPSGraphTensor * _Nonnull) coordinates
                                                  layout:(MPSGraphTensorNamedDataLayout) layout
                                    normalizeCoordinates:(BOOL) normalizeCoordinates
                                     relativeCoordinates:(BOOL) relativeCoordinates
                                            alignCorners:(BOOL) alignCorners
                                             paddingMode:(MPSGraphPaddingMode) paddingMode
                                     nearestRoundingMode:(MPSGraphResizeNearestRoundingMode_Ventura) nearestRoundingMode
                                           constantValue:(double) constantValue
                                                    name:(NSString * _Nullable) name;
- (MPSGraphTensor * _Nonnull) truncateWithTensor:(MPSGraphTensor * _Nonnull) tensor
                                            name:(NSString * _Nullable) name;

@end
