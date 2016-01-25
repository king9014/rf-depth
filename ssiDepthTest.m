function [ ] = ssiDepthTest( imPath,model )
% read test images
imgIds=dir(imPath); imgIds=imgIds([imgIds.bytes]>0);
imgIds={imgIds.name}; ext=imgIds{1}(end-2:end);
nImgs=length(imgIds); 
if(strcmpi(model.opts.dataSet,'make3d')), for i=1:nImgs, imgIds{i}=imgIds{i}(5:end-4); end, end
if(strcmpi(model.opts.dataSet,'nyu')), 
    for i=1:nImgs, imgIds{i}=imgIds{i}(1:end-4); end
    nyudat=load('nyudat/nyuTrainTest.mat'); nImgs=length(nyudat.testset);
end

outDir=['outfolder/' model.opts.dataSet];
if(~exist(outDir,'dir')), mkdir(outDir); end

for i=1:nImgs
    if(strcmpi(model.opts.dataSet,'make3d'))
        im = imread([imPath 'img-' imgIds{i} '.' ext]);
        depth = ssiDepthDetect(im, model);
        save([outDir '/' imgIds{i} '.mat'], 'depth');
    end
    if(strcmpi(model.opts.dataSet,'nyu'))
        im = imread([imPath imgIds{nyudat.testset(i)} '.' ext]);
        depth = ssiDepthDetect(im, model);
        save([outDir '/' imgIds{nyudat.testset(i)} '.mat'], 'depth');
    end
end

end

