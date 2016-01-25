dataSet='make3d';

if(strcmpi(dataSet,'make3d')), ssiDepthPath='outfolder/make3d/'; gtDepthPath='D:/Downloads/Test134Depth/';
    imPath='D:/Downloads/Test134Im/'; end
if(strcmpi(dataSet,'nyu')), ssiDepthPath='outfolder/nyu/'; gtDepthPath='D:/Downloads/NyuDepth/';
    imPath='D:/Downloads/NyuIm/'; end

imgIds=dir(ssiDepthPath); imgIds=imgIds([imgIds.bytes]>0);
imgIds={imgIds.name}; ext=imgIds{1}(end-2:end);
if(strcmpi(dataSet,'make3d')), nImgs=length(imgIds); end
if(strcmpi(dataSet,'nyu')), nyudat=load('nyudat/nyuTrainTest.mat'); nImgs=length(nyudat.testset); end
for i=1:nImgs, imgIds{i}=imgIds{i}(1:end-4); end

for i=1:nImgs
    ssi=load([ssiDepthPath imgIds{i} '.mat']); ssiDepth=ssi.depth;
    if(strcmpi(dataSet,'make3d')), 
        gt=load([gtDepthPath 'depth_sph_corr-' imgIds{i} '.mat']); 
        laserDepth=gt.Position3DGrid(:,:,4);
        I=imread([imPath 'img-' imgIds{i} '.jpg']); I=imresize(I,[400 300]); 
    end
    if(strcmpi(dataSet,'nyu')),
        gt=load([gtDepthPath imgIds{nyudat.testset(i)} '.mat']); laserDepth=gt.depth;
        I=imread([imPath imgIds{nyudat.testset(i)} '.jpg']); I=imresize(I,[400 300]); 
    end
    laserDepth=imresize(laserDepth,size(ssiDepth));
    imagesc([laserDepth ssiDepth]); axis off, colormap('jet')
    saveas(gcf,'depth.jpg');
    depth=imread('depth.jpg'); depth=depth(68:801,158:1086,:); depth=imresize(depth,[400 600]);
    if(strcmpi(dataSet,'make3d')), 
        imwrite([I depth], ['outfolder/make3d_plot/' imgIds{i} '.jpg'], 'jpg'); end
    if(strcmpi(dataSet,'nyu')),
        imwrite([I depth], ['outfolder/nyu_plot/' imgIds{nyudat.testset(i)} '.jpg'], 'jpg'); end
end