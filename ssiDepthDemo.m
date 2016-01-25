% Demo for Single Still Image Depth Detector (please see readme.txt first).
addpath(genpath('toolbox'));
addpath(genpath('features'));
rootPath = 'D:/Downloads';

%% set opts for training (see ssiDepthTrain.m)
opts=ssiDepthTrain();             % default options (good settings)
opts.modelFnm='modelMake3d';     % model name
opts.dataSet='make3d';
opts.trainImDir=[rootPath '/Train400Im/'];
opts.gtMatDir=[rootPath '/Train400Depth/'];
opts.useParfor=0;                 % parallelize if sufficient memory

%% train depth detector
tic, model=ssiDepthTrain(opts); toc; % will load model if already trained

%% set detection parameters (can set after training)
model.opts.nTreesEval=4;          % for top speed set nTreesEval=1
model.opts.nThreads=4;            % max number threads for evaluation

%% evaluate detector
if(0), 
if(strcmpi(model.opts.dataSet,'make3d')), ssiDepthTest([rootPath '/Test134Im/'], model); 
    [rel,lg10,rmse]=ssiDepthEval([rootPath '/Test134Im/'],[rootPath '/Test134Depth/'],model);
end
if(strcmpi(model.opts.dataSet,'nyu')), ssiDepthTest([rootPath '/NyuIm/'], model); 
    [rel,lg10,rmse]=ssiDepthEval([rootPath '/NyuIm/'],[rootPath '/NyuDepth/'],model);
end
save([opts.modelFnm '_eval.mat'],'rel','lg10','rmse');
end

if(1),
I=imread('papper.jpg');
tic, depth=ssiDepthDetect(I, model); toc;
subplot(1,2,1);
imshow(I);
subplot(1,2,2);
imagesc(depth),colorbar;
end