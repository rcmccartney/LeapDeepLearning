function [net, info] = cnn_leap(varargin)
% CNN_MNIST  Demonstrated MatConNet on MNIST

run(fullfile('C:\Users\mccar_000\SkyDrive\git\matconvnet-1.0-beta9', ...
    'matlab', 'vl_setupnn.m')) ;

addpath(fullfile('C:\Users\mccar_000\SkyDrive\git\matconvnet-1.0-beta9', ...
    'examples'));

opts.dataDir = fullfile('data','data') ;
opts.expDir = fullfile('data','leap-oneimage') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train.batchSize = 100 ;
opts.train.numEpochs = 2000 ;
opts.train.continue = true ;
opts.train.useGpu = false ;
opts.train.learningRate = 0.001 ;
opts.train.expDir = opts.expDir ;
opts = vl_argparse(opts, varargin) ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% Define a network similar to LeNet
f= 1/100;
net.layers = {} ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(5,5,1,20, 'single'), ...
                           'biases', zeros(1, 20, 'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(5,5,20,50, 'single'),...
                           'biases', zeros(1,50,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(4,4,50,500, 'single'),...
                           'biases', zeros(1,500,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(1,1,500,12, 'single'),...
                           'biases', zeros(1,12,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

% Take the mean out and make GPU if needed
if opts.train.useGpu
  imdb.images.data = gpuArray(imdb.images.data) ;
end

[net, info] = cnn_train(net, imdb, @getBatch, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

% --------------------------------------------------------------------
function imdb = getImdb(opts)
% --------------------------------------------------------------------

gestures =  { {'capE'}, {'CheckMark'}, {'e'}, {'F'}, {'Figure8'}, {'Grab'}, ... 
    {'Pinch'}, {'Release'}, {'Swipe'}, {'Tap'}, {'Tap2'}, {'Wipe'} };

% Prepare the imdb structure, returns image data with mean image subtracted
if ~exist(opts.dataDir, 'dir')
  mkdir(opts.dataDir) ;
end

% specific to the dataset
top = 'LeapData';
linesToSkip = 2;
dim = 200;
sampleSize = 50; 
%set up the 15 images to use (5 fingers x 3 planes)
numfingers = 1; % 5
numdims = 1; % 2 is XY, XZ; 3 adds YZ
testfrac = 0.9;

% get all the directories & remove '.' & '..'
files = dir(top);
directoryNames = {files([files.isdir]).name};
directoryNames = directoryNames(~ismember(directoryNames,{'.','..'}));
% loop over each gesture directory
index = 1;
for i=1:length(directoryNames),
    aviobj = VideoWriter(fullfile('videos', strcat(directoryNames{i}, '.avi')));
    open(aviobj);
    sprintf('#### Starting %s ####',directoryNames{i})
    folders = dir(fullfile(top, directoryNames{i}));
    folderNames = {folders([folders.isdir]).name};
    folderNames = folderNames(~ismember(folderNames,{'.','..'}));
    % now folders 
    for j=1:length(folderNames),
        files = dir(fullfile(top, directoryNames{i}, folderNames{j}));
        fileNames = {files(~[files.isdir]).name};
        for k=1:length(fileNames),
            image = ImageMaker_old(fullfile(top, directoryNames{i}, ...
                        folderNames{j}, fileNames{k}), ...
                        dim, sampleSize, linesToSkip, numfingers, numdims);
            imshow(image)   
            writeVideo(aviobj,image);    
            if ~isempty(image),
                images(:,:,index) = image;
                % this is the output class, 1 through 12directoryNames{i}
                output(index) = find(strcmp([gestures{:}], directoryNames{i}));
                index = index + 1;        
            end;
        end;
    end;
    close(aviobj);
end;

% mix up the classes 
shuffle = randperm(size(output,2));
% images is a 3D matrix that is (28, 28*3, numimages) large
images = images(:,:,shuffle);
% output is one row, with columns of class labels 1-12 
output = output(shuffle);
% split into train and test
trainsize = int64(testfrac*size(output,2));
testsize = size(output,2) - trainsize;

% set is a row of ones then threes used by library for training and test sets
% a two would be validation set, not used here
set = [ones(1,trainsize) 3*ones(1,testsize)];
% added a space for convolutions
data = single(reshape(images,sampleSize*numfingers,sampleSize*numdims,1,[]));

% get the mean of each image so it can be subtracted
% taking this out for now
dataMean = mean(data(:,:,:,set == 1), 4);
% data = bsxfun(@minus, data, dataMean) ;

imdb.images.data = data;
imdb.images.data_mean = dataMean;
imdb.images.labels = output;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'};
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),1:12,'uniformoutput',false); 
