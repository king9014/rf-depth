function model = ssiDepthTrain( varargin )
% Train structured depth detector.
%
% For an introductory tutorial please see ssiDepthDemo.m.
%
% USAGE
%  opts = ssiDepthTrain()
%  model = ssiDepthTrain( opts )
%
% INPUTS
%  opts       - parameters (struct or name/value pairs)
%   (1) model parameters:
%   .imWidth    - [32] width of image patches
%   .gtWidth    - [16] width of ground truth patches
%   (2) tree parameters:
%   .nSamp      - [5e5] number of sampled patches per tree
%   .nImgs      - [inf] maximum number of images to use for training
%   .nTrees     - [8] number of trees in forest to train
%   .fracFtrs   - [1/4] fraction of features to use to train each tree
%   .minCount   - [1] minimum number of data points to allow split
%   .minChild   - [8] minimum number of data points allowed at child nodes
%   .maxDepth   - [64] maximum depth of tree
%   .discretize - ['pca'] options include 'pca' and 'kmeans'
%   .nClasses   - [2] number of classes (clusters) for binary splits
%   .split      - ['gini'] options include 'gini', 'entropy' and 'twoing'
%   (3) feature parameters:
%   .chnSmooth  - [2] radius for reg channel smoothing (using convTri)
%   .simSmooth  - [8] radius for sim channel smoothing (using convTri)
%   .normRad    - [4] gradient normalization radius (see gradientMag)
%   .shrink     - [2] amount to shrink channels
%   .nCells     - [5] number of self similarity cells
%   (4) detection parameters (can be altered after training):
%   .stride     - [2] stride at which to compute depth
%   .nTreesEval - [4] number of trees to evaluate per location
%   .nThreads   - [4] number of threads for evaluation of trees
%   (5) other parameters:
%   .seed       - [1] seed for random stream (for reproducibility)
%   .useParfor  - [0] if true train trees in parallel (memory intensive)
%   .modelDir   - ['models/'] target directory for storing models
%   .modelFnm   - ['model'] model filenamet
%
% OUTPUTS
%  model      - trained structured depth detector w the following fields
%   .opts       - input parameters and constants
%   .thrs       - [nNodes x nTrees] threshold corresponding to each fid
%   .fids       - [nNodes x nTrees] feature ids for each node
%   .child      - [nNodes x nTrees] index of child for each node
%   .count      - [nNodes x nTrees] number of data points at each node
%   .depth      - [nNodes x nTrees] depth of each node
%
% EXAMPLE
%
% See also ssiDepthDemo, ssiDepthChns, ssiDepthDetect, forestTrain
%
% Structured Depth Estimation Toolbox      Version 1.0
% Code written by Ren Jin, 2016.

% get default parameters
dfs={'imWidth',32, 'gtWidth',16, 'nSamp',5e5, 'nImgs',inf, ...
  'nTrees',8, 'fracFtrs',1/4, 'minCount',1, 'minChild',4, ...
  'maxDepth',64, 'discretize','pca', 'nClasses',2, 'split','gini', ...
  'chnSmooth',2, 'simSmooth',8, 'shrink',2,'shrinkCol',8, ...
  'nCells',5,'nCellsCol',2, 'stride',2, 'nTreesEval',4, ...
  'nThreads',4, 'seed',1, 'useParfor',0, 'modelDir','models/', ...
  'modelFnm','model', 'imResize',[256 336], 'trainImDir','D:/Downloads/Train400Im/', ...
  'gtMatDir','D:/Downloads/Train400Depth/','dataSet','make3d','refine',0};
opts = getPrmDflt(varargin,dfs,1);
if(nargin==0), model=opts; return; end

% if forest exists load it and return
forestDir = [opts.modelDir '/forest/'];
forestFn = [forestDir opts.modelFnm];
if(exist([forestFn '.mat'], 'file'))
  load([forestFn '.mat']); return; end

% compute constants and store in opts
nTrees=opts.nTrees; trnImgDir=opts.trainImDir;
imgIds=dir(trnImgDir); imgIds=imgIds([imgIds.bytes]>0);
imgIds={imgIds.name}; I=imread([trnImgDir imgIds{1}]); opts.nChns=1;
[~,~,~,~,nChnFtrs,nColChnFtrs,nSimFtrs,nColSimFtrs,nChns]=ssiDepthChns(I, opts);
opts.nChns=nChns; opts.nChnFtrs=nChnFtrs; opts.nColChnFtrs=nColChnFtrs;
opts.nSimFtrs=nSimFtrs; opts.nColSimFtrs=nColSimFtrs;
opts.nTotFtrs=nChnFtrs+nColChnFtrs+nSimFtrs+nColSimFtrs; disp(opts);

% generate stream for reproducibility of model
stream=RandStream('mrg32k3a','Seed',opts.seed);

% train nTrees random trees (can be trained with parfor if enough memory)
if(opts.useParfor), parfor i=1:nTrees, trainTree(opts,stream,i); end
else for i=1:nTrees, trainTree(opts,stream,i); end; end

% merge trees and save model
model = mergeTrees( opts );
if(~exist(forestDir,'dir')), mkdir(forestDir); end
save([forestFn '.mat'], 'model', '-v7.3');

end

function model = mergeTrees( opts )
% accumulate trees and merge into final model
nTrees=opts.nTrees; gtWidth=opts.gtWidth;
treeFn = [opts.modelDir '/tree/' opts.modelFnm '_tree'];
for i=1:nTrees
  t=load([treeFn int2str2(i,3) '.mat'],'tree'); t=t.tree;
  if(i==1), trees=t(ones(1,nTrees)); else trees(i)=t; end
end
nNodes=0; for i=1:nTrees, nNodes=max(nNodes,size(trees(i).fids,1)); end
% merge all fields of all trees
model.opts=opts; Z=zeros(nNodes,nTrees,'uint32');
model.thrs=zeros(nNodes,nTrees,'single');
model.fids=Z; model.child=Z; model.count=Z; model.depth=Z;
model.segs=zeros(gtWidth,gtWidth,nNodes,nTrees,'uint8');
for i=1:nTrees, tree=trees(i); nNodes1=size(tree.fids,1);
  model.fids(1:nNodes1,i) = tree.fids;
  model.thrs(1:nNodes1,i) = tree.thrs;
  model.child(1:nNodes1,i) = tree.child;
  model.count(1:nNodes1,i) = tree.count;
  model.depth(1:nNodes1,i) = tree.depth;
  model.segs(:,:,1:nNodes1,i) = tree.hs;
end
end

function trainTree( opts, stream, treeInd )
% Train a single tree in forest model.

% location of ground truth
trnImgDir = opts.trainImDir;
trnGtDir = opts.gtMatDir;
imgIds=dir(trnImgDir); imgIds=imgIds([imgIds.bytes]>0);
imgIds={imgIds.name}; ext=imgIds{1}(end-2:end);
nImgs=length(imgIds); 
if(strcmpi(opts.dataSet,'make3d'))
    for i=1:nImgs, imgIds{i}=imgIds{i}(4:end-4); end
end
if(strcmpi(opts.dataSet,'nyu'))
    for i=1:nImgs, imgIds{i}=imgIds{i}(1:end-4); end
end

% extract commonly used options
imWidth=opts.imWidth; % imRadius=imWidth/2;
gtWidth=opts.gtWidth; % gtRadius=gtWidth/2;
nChns=opts.nChns; nTotFtrs=opts.nTotFtrs;
nPatch=opts.nSamp; shrink=opts.shrink; shrinkCol=opts.shrinkCol;
gtShrink=imWidth/gtWidth; imResize=opts.imResize;

% finalize setup
treeDir = [opts.modelDir '/tree/'];
treeFn = [treeDir opts.modelFnm '_tree'];
if(exist([treeFn int2str2(treeInd,3) '.mat'],'file'))
  fprintf('Reusing tree %d of %d\n',treeInd,opts.nTrees); return; end
fprintf('\n-------------------------------------------\n');
fprintf('Training tree %d of %d\n',treeInd,opts.nTrees); tStart=clock;

% set global stream to stream with given substream (will undo at end)
streamOrig = RandStream.getGlobalStream();
set(stream,'Substream',treeInd);
RandStream.setGlobalStream( stream );

% collect patches and compute features
fids=sort(randperm(nTotFtrs,round(nTotFtrs*opts.fracFtrs)));
k = nPatch; nImgs=min(nImgs,opts.nImgs);
ftrs = zeros(k,length(fids),'single');
labels = zeros(gtWidth,gtWidth,k,'uint8'); k = 0;
tid = ticStatus('Collecting data',30,1);
if(strcmpi(opts.dataSet,'nyu')), nyudat=load('nyudat/nyuTrainTest.mat'); nImgs=length(nyudat.trainset); end
for i = 1:nImgs
    % get image and compute channels
    if(strcmpi(opts.dataSet,'make3d'))
        gt=load([trnGtDir 'depth_sph_corr' imgIds{i} '.mat']);
        depth=uint8(round(gt.Position3DGrid(:,:,4)));
        I=imread([trnImgDir 'img' imgIds{i} '.' ext]); % siz=size(I);
    end
    if(strcmpi(opts.dataSet,'nyu'))
        gt=load([trnGtDir imgIds{nyudat.trainset(i)} '.mat']);
        depth=uint8(round(gt.depth*10));
        I=imread([trnImgDir imgIds{nyudat.trainset(i)} '.' ext]);
    end
    % p=zeros(1,4); p([2 4])=mod(4-mod(siz(1:2),4),4);
    % if(any(p)), I=imPad(I,p,'symmetric'); end
    [chnsReg,colsReg,chnsSim,colsSim] = ssiDepthChns(I,opts);
    gtSiz=imResize/gtShrink; depth=imresize(depth,gtSiz);
    % 计算行列 采样数目
    k1=ceil(nPatch/nImgs);
    nSamRow=floor(sqrt(k1/(gtSiz(1)/gtSiz(2))));
    % nSamRow=floor(sqrt(k1/(imResize(1)/imResize(2))));
    nSamCol=floor(sqrt(k1/(gtSiz(2)/gtSiz(1))));
    % nSamCol=floor(sqrt(k1/(imResize(2)/imResize(1))));
    nStepCol=ceil((gtSiz(2)-gtWidth)/nSamRow);
    % nStepCol=ceil((imResize(2)-imWidth)/nSamRow);
    nStepRow=ceil((gtSiz(1)-gtWidth)/nSamCol);
    % nStepRow=ceil((imResize(1)-imWidth)/nSamCol);
    % crop patches and ground truth labels
    psReg=zeros(imWidth/shrink,imWidth/shrink,nChns,k1,'single');
    psColReg=zeros(imResize(1)/shrinkCol,imWidth/shrinkCol,nChns,k1,'single');
    lbls=zeros(gtWidth,gtWidth,k1,'uint8');
    psSim=psReg; psColSim=psColReg; k1=0;
    sRadiusR=ceil(nStepRow/2); sRadiusC=ceil(nStepCol/2); % 信息熵搜索半径
    for posR=1:nStepRow:gtSiz(1)-gtWidth
        for posC=1:nStepCol:gtSiz(2)-gtWidth
            k1=k1+1; tr=posR; tc=posC; tf=1e16;
            if(0), for sr=posR:posR+sRadiusR
                for sc=posC:posC+sRadiusC
                    if sr>gtSiz(1)-gtWidth || sc>gtSiz(2)-gtWidth, continue; end
                    dp=depth(sr:sr+gtWidth-1,sc:sc+gtWidth-1);
                    f=sum(sum(dp.*log(dp)./log(2))); % f=-sum(sum(dp.*log(dp)./log(2)));
                    if f<tf, tf=f; tr=sr; tc=sc; end
                end
            end
            end
            lbls(:,:,k1)=depth(tr:tr+gtWidth-1,tc:tc+gtWidth-1);
            posFR=ceil(tr*gtShrink/shrink); posFC=ceil(tc*gtShrink/shrink); ftWidth=imWidth/shrink;
            colFC=ceil(tc*gtShrink/shrinkCol); colWidth=imWidth/shrinkCol;
            psReg(:,:,:,k1)=chnsReg(posFR:posFR+ftWidth-1,posFC:posFC+ftWidth-1,:);
            psSim(:,:,:,k1)=chnsSim(posFR:posFR+ftWidth-1,posFC:posFC+ftWidth-1,:);
            psColReg(:,:,:,k1)=colsReg(:,colFC:colFC+colWidth-1,:);
            psColSim(:,:,:,k1)=colsSim(:,colFC:colFC+colWidth-1,:);
        end
    end
    if(0), figure(1); montage2(squeeze(psReg(:,:,1,:))); drawnow; end
    if(0), figure(2); montage2(lbls(:,:,:)); drawnow; end
    % compute features and store
    psReg=psReg(:,:,:,1:k1); psSim=psSim(:,:,:,1:k1); lbls=lbls(:,:,1:k1);
    psColReg=psColReg(:,:,:,1:k1); psColSim=psColSim(:,:,:,1:k1);
    ftrs1=[reshape(psReg,[],k1)' reshape(psColReg,[],k1)' stComputeSimFtrs(psSim,opts)  stComputeSimFtrsCol(psColSim,opts)];
    ftrs(k+1:k+k1,:)=ftrs1(:,fids); labels(:,:,k+1:k+k1)=lbls;
    k=k+k1; if(k==size(ftrs,1)), tocStatus(tid,1); break; end
    tocStatus(tid,i/nImgs);
end
if(k<size(ftrs,1)), ftrs=ftrs(1:k,:); labels=labels(:,:,1:k); end

% train structured edge classifier (random decision tree)
pTree=struct('minCount',opts.minCount, 'minChild',opts.minChild, ...
  'maxDepth',opts.maxDepth, 'H',opts.nClasses, 'split',opts.split);
t=labels; labels=cell(k,1); for i=1:k, labels{i}=t(:,:,i); end
pTree.discretize=@(hs,H) discretize(hs,H,opts.discretize);
tree=forestTrain(ftrs,labels,pTree); tree.hs=cell2array(tree.hs);
tree.fids(tree.child>0) = fids(tree.fids(tree.child>0)+1)-1;
if(~exist(treeDir,'dir')), mkdir(treeDir); end
save([treeFn int2str2(treeInd,3) '.mat'],'tree'); e=etime(clock,tStart);
fprintf('Training of tree %d complete (time=%.1fs).\n',treeInd,e);
RandStream.setGlobalStream( streamOrig );

end

function ftrs = stComputeSimFtrsCol( chns, opts )
% Compute self-similarity features (order must be compatible w mex file).
w1=opts.imResize(1)/opts.shrinkCol; w2=opts.imWidth/opts.shrinkCol; 
nPatchProp=opts.imResize(1)/opts.imWidth; n2=opts.nCellsCol; n1=n2*nPatchProp;
if(n2==0), ftrs=[]; return; end
nColSimFtrs=opts.nColSimFtrs; nChns=opts.nChns; m=size(chns,4);
inds1=round(w1/n1/2); inds1=round((1:n1)*(w1+2*inds1-1)/(n1+1)-inds1+1);
inds2=round(w2/n2/2); inds2=round((1:n2)*(w2+2*inds2-1)/(n2+1)-inds2+1);
chns=reshape(chns(inds1,inds2,:,:),n1*n2,nChns,m);
ftrs=zeros(nColSimFtrs/nChns,nChns,m,'single');
k=0; for i=1:n1*n2-1, k1=n1*n2-i; i1=ones(1,k1)*i;
  ftrs(k+1:k+k1,:,:)=chns(i1,:,:)-chns((1:k1)+i,:,:); k=k+k1; end
ftrs = reshape(ftrs,nColSimFtrs,m)';
end

function ftrs = stComputeSimFtrs( chns, opts )
% Compute self-similarity features (order must be compatible w mex file).
w=opts.imWidth/opts.shrink; n=opts.nCells; if(n==0), ftrs=[]; return; end
nSimFtrs=opts.nSimFtrs; nChns=opts.nChns; m=size(chns,4);
inds=round(w/n/2); inds=round((1:n)*(w+2*inds-1)/(n+1)-inds+1);
chns=reshape(chns(inds,inds,:,:),n*n,nChns,m);
ftrs=zeros(nSimFtrs/nChns,nChns,m,'single');
k=0; for i=1:n*n-1, k1=n*n-i; i1=ones(1,k1)*i;
  ftrs(k+1:k+k1,:,:)=chns(i1,:,:)-chns((1:k1)+i,:,:); k=k+k1; end
ftrs = reshape(ftrs,nSimFtrs,m)';
end

function [hs,segs] = discretize( segs, nClasses, type )
zs=cell2array(segs); n=length(segs);
zs=reshape(zs,[], n)';
zs=double(zs)/255;
zs=bsxfun(@minus,zs,sum(zs,1)/n); 
zs=zs(:,any(zs,1));
if(isempty(zs)), hs=ones(n,1,'uint32'); segs=segs{1}; return; end
% find most representative segs (closest to mean)
[~,ind]=min(sum(zs.*zs,2)); segs=segs{ind};
if(strcmpi(type,'pca'))
    % apply PCA to reduce dimensionality of zs
    U=pca(zs'); d=min(5,size(U,2)); zs=zs*U(:,1:d);
    % discretize zs by clustering or discretizing pca dimensions
    d=min(d,floor(log2(nClasses))); hs=zeros(n,1);
    for i=1:d, hs=hs+(zs(:,i)<0)*2^(i-1); end
    [~,~,hs]=unique(hs); 
end
if(strcmpi(type,'kmeans'))
    [~,hs] = vl_kmeans(zs',nClasses);
end
if(strcmpi(type,'gmm'))
    [~,~,~,~,posteriors] = vl_gmm(zs', nClasses);
    hs=zeros(n,1);
    for i=1:n, [~,Yi]=max(posteriors(:,i)); 
        hs(i)=Yi; end
end
if(strcmpi(type,'fcm'))
    % U=pca(zs'); d=min(16,size(U,2)); zs=zs*U(:,1:d);
    [~,U,~] = fcm(zs, nClasses);
    hs=zeros(n,1);
    for i=1:n, [~,Yi]=max(U(:,i)); 
        hs(i)=Yi; end
end
hs=uint32(hs);
% optionally display different types of hs
for i=1:0, figure(i); montage2(cell2array(segs(hs==i))); end
end
