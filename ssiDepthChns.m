function [chnsReg,colsReg,chnsSim,colsSim,nChnFtrs,nColChnFtrs,nSimFtrs,nColSimFtrs,nChns] = ssiDepthChns( I, opts )
% Compute features for structured depth detection.
%
% For an introductory tutorial please see ssiDepthDemo.m.
%
% USAGE
%  [chnsReg,chnsSim] = ssiDepthChns( I, opts )
%
% INPUTS
%  I          - [h x w x 3] color input image
%  opts       - structured depth model options
%
% OUTPUTS
%  chnsReg    - [h x w x nChannel] regular output channels
%  chnsSim    - [h x w x nChannel] self-similarity output channels
%
% EXAMPLE
%
% See also ssiDepthDemo, ssiDepthChns, ssiDepthDetect, forestTrain
%
% Structured Depth Estimation Toolbox      Version 1.0
% Code written by Ren Jin, 2016.

resSiz=opts.imResize;
I=imresize(I,resSiz,'nearest'); I=single(I)./255;
% sample parameter
shrink=opts.shrink;  shrinkCol=opts.shrinkCol; chns=cell(1,opts.nChns); k=0;
% grdSmooth=0; normRad=4; nOrients=4; 
chnSmooth=opts.chnSmooth; simSmooth=opts.simSmooth;
sigFtrSiz=(opts.imWidth/shrink).^2;
colFtrSiz=[resSiz(1) opts.imWidth]/shrinkCol; colFtrSiz=colFtrSiz(1)*colFtrSiz(2);

%% multi sample start
Irgb=I; IrgbShrink=imResample(Irgb,1/shrink); % Jario Add Rgb Ftrs
Iluv=rgbConvert(I,'luv'); IluvShrink=imResample(Iluv,1/shrink);
% Ihsv=rgbConvert(I,'hsv'); IhsvShrink=imResample(Ihsv,1/shrink);
Ihsi=single(rgb2hsi(double(I))); IhsiShrink=imResample(Ihsi,1/shrink);
prior=zeros(resSiz/shrink, 'single');
for i=1:size(prior,1), prior(i,:)=i/size(prior,1); end
k=k+1; chns{k}=prior; % k=k+1; chns{k}=prior; k=k+1; chns{k}=prior;
k=k+1; chns{k}=IrgbShrink; k=k+1; chns{k}=IhsiShrink; k=k+1; chns{k}=IluvShrink;
for i = 1:2, s=2^(i-1);
    % if(s==shrink), I1=Ishrink; else I1=imResample(I,1/s); end
    if(s==shrink), I2=IrgbShrink; else I2=imResample(Irgb,1/s); end
    % Jario Add Start
    S1H1=single(calculateFilterBanks_old(double(I2))); 
    % tic, dark=single(darkChannel(double(I2))); toc;
    dark=ssiDarkChannelMex(I2);
    k=k+1; chns{k}=imResample(S1H1,s/shrink);
    k=k+1; chns{k}=imResample(dark,s/shrink);
    % Jario End
%     I1 = convTri( I1, grdSmooth );
%     [M,O] = gradientMag( I1, 0, normRad, .01 );
%     H = gradientHist( M, O, max(1,shrink/s), nOrients, 0 );
%     k=k+1; chns{k}=imResample(M,s/shrink);
%     k=k+1; chns{k}=imResample(H,max(1,s/shrink));
end


%% sample finished 
chns=cat(3,chns{1:k}); % assert(size(chns,3)==opts.nChns);
chnSm=chnSmooth/shrink; if(chnSm>1), chnSm=round(chnSm); end
simSm=simSmooth/shrink; if(simSm>1), simSm=round(simSm); end
chnsReg=convTri(chns,chnSm); chnsSim=convTri(chns,simSm);
% chnsCol=imResample(chnsReg,1/shrinkCol*shrink);
colsReg=imresize(chnsReg,floor(resSiz/shrinkCol),'nearest');
colsSim=imresize(chnsSim,floor(resSiz/shrinkCol),'nearest');

nCells=opts.nCells; nCellsCol=opts.nCellsCol; nChns=size(chnsReg,3);
nChnFtrs=sigFtrSiz*nChns;
nColChnFtrs=colFtrSiz*nChns;
nPatchProp=resSiz(1)/opts.imWidth; nVertical=nCellsCol*nPatchProp;
nColSimFtrs=(nVertical*nCellsCol)*(nVertical*nCellsCol-1)/2*nChns;
nSimFtrs=(nCells*nCells)*(nCells*nCells-1)/2*nChns;
end
