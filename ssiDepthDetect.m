function [depthMap] = ssiDepthDetect( I, model )
% Detect depth in image.
%
% For an introductory tutorial please see ssiDepthDemo.m.
%
% USAGE
%  [depthMap] = ssiDepthDetect( I, model )
%
% INPUTS
%  I          - [h x w x 3] color input image
%  model      - structured depth model trained with ssiDepthTrain
%
% OUTPUTS
%  depthMap   - [h x w] depth probability map
%
% EXAMPLE
%
% See also ssiDepthDemo, ssiDepthChns, ssiDepthDetect, forestTrain
%
% Structured Depth Estimation Toolbox      Version 1.0
% Code written by Ren Jin, 2016.

% get parameters
opts=model.opts; opts.nTreesEval=min(opts.nTreesEval,opts.nTrees);
opts.stride=max(opts.stride,opts.shrink); model.opts=opts;

% compute features and apply forest to image
% nPixelPredict=(opts.gtWidth.^2)*opts.nTreesEval;

[chnsReg,colsReg,chnsSim,colsSim] = ssiDepthChns( I, opts );
I=imresize(I, opts.imResize);
[depthMap] = ssiDepthDetectMex(model,I,chnsReg,colsReg,chnsSim,colsSim);
depthMap=convTri(depthMap,4);


end
