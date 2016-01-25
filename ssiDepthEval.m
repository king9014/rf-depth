function [ rel,lg10,rmse ] = ssiDepthEval( depthPath,gtPath,model )
% read test images

imgIds=dir(depthPath); imgIds=imgIds([imgIds.bytes]>0);
imgIds={imgIds.name}; % ext=imgIds{1}(end-2:end);
nImgs=length(imgIds); 
if(strcmpi(model.opts.dataSet,'make3d')),
    for i=1:nImgs, imgIds{i}=imgIds{i}(5:end-4); end
end
if(strcmpi(model.opts.dataSet,'nyu')), 
    for i=1:nImgs, imgIds{i}=imgIds{i}(1:end-4); end
    nyudat=load('nyudat/nyuTrainTest.mat'); nImgs=length(nyudat.testset);
end
l=1; 
% Run structured depth
fprintf(['\nRunning structured depth on all test images in ',model.opts.dataSet,'\n', ...
           '=========================================\n']);
for i=1:nImgs
    if(strcmpi(model.opts.dataSet,'make3d')),
        ssiDepth=load(['outfolder/' model.opts.dataSet '/' imgIds{i} '.mat']);
        depthMap=ssiDepth.depth;
        gt=load([gtPath 'depth_sph_corr-' imgIds{i} '.mat']); laserdepth=gt.Position3DGrid(:,:,4);
        depthMap=imresize(depthMap,size(laserdepth)); laserdepth(laserdepth>80)=80;
    end
    if(strcmpi(model.opts.dataSet,'nyu')),
        ssiDepth=load(['outfolder/' model.opts.dataSet '/' imgIds{nyudat.testset(i)} '.mat']);
        depthMap=ssiDepth.depth;
        gt=load([gtPath imgIds{nyudat.testset(i)} '.mat']); laserdepth=gt.depth;
        depthMap=imresize(depthMap,size(laserdepth));
    end
    
    if(strcmpi(model.opts.dataSet,'make3d')),
        C1=laserdepth<70;
        rel_C1 = abs( (depthMap(C1)-laserdepth(C1))./laserdepth(C1) );
        lg10_C1 = abs( log10(depthMap(C1))-log10(laserdepth(C1)) );
        rmse_C1 = abs( (depthMap(C1))-(laserdepth(C1)) );
        whole_rel_C1{l} = rel_C1;
        whole_lg10_C1{l} = lg10_C1;
        whole_rmse_C1{l} = rmse_C1;
    end
    
    rel_C2 = abs( (depthMap(:)-laserdepth(:))./laserdepth(:) );
    lg10_C2 = abs( log10(depthMap(:))-log10(laserdepth(:)) );
    rmse_C2 = abs( (depthMap(:))-(laserdepth(:)) );
    whole_rel_C2{l} = rel_C2;
    whole_lg10_C2{l} = lg10_C2;
    whole_rmse_C2{l} = rmse_C2;
    
    l = l+1;
end

rel = mean(cell2mat(whole_rel_C2'));
lg10 = mean(cell2mat(whole_lg10_C2'));
rmse = sqrt(mean(cell2mat(whole_rmse_C2').^2));

fprintf(['\n',model.opts.dataSet,' Error averaged over all test data\n', ...
           '=================================\n', ...
           '%10s %10s %10s\n%10.3f %10.3f %10.3f\n'], ...
           'relative', 'log10', 'rmse', rel, lg10, rmse);

if(strcmpi(model.opts.dataSet,'make3d')),
    relC1 = mean(cell2mat(whole_rel_C1'));
    lg10C1 = mean(cell2mat(whole_lg10_C1'));
    rmseC1 = sqrt(mean(cell2mat(whole_rmse_C1').^2));

    fprintf(['\nMake3D C1 Error averaged over all test data\n', ...
               '=================================\n', ...
               '%10s %10s %10s\n%10.3f %10.3f %10.3f\n'], ...
               'relative', 'log10', 'rmse', relC1, lg10C1, rmseC1);
end
end

