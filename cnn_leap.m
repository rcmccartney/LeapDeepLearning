function [net, info] = cnn_leap(varargin)
% Here activated pixels from the Leap device are treated as a binary 0/1

%run(fullfile('C:\Users\Henry\Box Sync\Projects\matconvnet-master', ...
%    'matlab', 'vl_setupnn.m')) ;

opts.dataDir = fullfile('data');
opts.expDir = fullfile('data', 'leap-binary-full');
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train.batchSize = 200 ;
opts.train.numEpochs = 100 ;  
opts.train.continue = true ;  % can continue training after stopping
opts.train.useGpu = true ;
opts.train.learningRate = [0.1*ones(1, 10) 0.01*ones(1, 25) 0.001*ones(1, 50) 0.0001*ones(1,50)  0.00001*ones(1,20)] ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.90 ;
opts.train.outputClasses = 12 ;
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
net.layers = {};
% First convolutional layer, 5x5 filters with a bias
% the images are 50x50
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
                           'filters', f*randn(6,6,20,50, 'single'),...
                           'biases', zeros(1,50,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(9,9,50,500, 'single'),...
                           'biases', zeros(1,500,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(1,1,500,opts.train.outputClasses, 'single'),...
                           'biases', zeros(1,opts.train.outputClasses ,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
tic ; 
% Take the mean out and make GPU if needed
if opts.train.useGpu
  imdb.images.data = gpuArray(imdb.images.data) ;
end

[net, info] = cnn_train(net, imdb, @getBatch, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;
toc
end

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
end

% --------------------------------------------------------------------
function imdb = getImdb(opts)
% --------------------------------------------------------------------

gestures =  { {'capE'}, {'CheckMark'}, {'e'}, {'F'}, {'Figure8'}, {'Swipe'},... 
              {'Tap'}, {'Grab'}, {'Release'}, {'Tap2'}, {'Wipe'}, {'Pinch'}};
excluded = { '.' , '..' } ;

% Prepare the imdb structure, returns image data with mean image subtracted
if ~exist(opts.dataDir, 'dir')
  mkdir(opts.dataDir) ;
end
if ~exist(opts.expDir, 'dir')
  mkdir(opts.expDir) ;
end

% specific to the dataset
top = 'LeapData';
linesToSkip = 2;
dim = 200;
sampleSize = 50 ; 
%set up the images to use (5 fingers x 3 planes)
numfingers = 5 ; 
numdims = 3 ;    % 2 is XY, XZ; 3 adds YZ
testfrac = 0.9;
makeVideo = true;
% this is the min number of pixels in image to be included in dataset
lowerbound = 100;  
% this gives % of total number of pixels, if more than this is activated
% the image is washed out
upperbound = 0.9 * sampleSize * sampleSize * numdims * numfingers;

% get all the directories & remove '.' & '..'
files = dir(top);
directoryNames = {files([files.isdir]).name};
directoryNames = directoryNames(~ismember(directoryNames, excluded));
% loop over each gesture directory
index = 1;
for i=1:length(directoryNames),
    if makeVideo,
        if ~exist(fullfile(opts.expDir, 'videos'), 'dir')
            mkdir(fullfile(opts.expDir, 'videos')) ;
        end
        aviobj = VideoWriter(fullfile(opts.expDir, 'videos', ...
        strcat(directoryNames{i}, '.avi')));
        open(aviobj);
    end;
    sprintf('#### Starting %s ####',directoryNames{i})
    folders = dir(fullfile(top, directoryNames{i}));
    folderNames = {folders([folders.isdir]).name};
    folderNames = folderNames(~ismember(folderNames,{'.','..'}));
    % now folders 
    for j=1:length(folderNames),
        files = dir(fullfile(top, directoryNames{i}, folderNames{j}));
        fileNames = {files(~[files.isdir]).name};
        for k=1:length(fileNames),
            image = ImageMaker_full(fullfile(top, directoryNames{i}, ...
                        folderNames{j}, fileNames{k}), ...
                        dim, sampleSize, linesToSkip, numfingers, numdims);
            % the image must have a minimum number of points to be incl.
            activated = nnz(image);
            if activated > lowerbound && activated < upperbound,
                imshow(image)   
                if makeVideo,
                    % need to make image in range 0 to 1
                    writeVideo(aviobj,mat2gray(image));    
                end;
                %str = input('Keep this image? (Enter-yes/N-no) ','s');
                % prompt user to keep an image or not
                %if strcmp('', str),
                    images(:,:,index) = image;
                    % this is the output class, 1 through 12directoryNames{i}
                    output(index) = find(strcmp([gestures{:}], directoryNames{i}));
                    index = index + 1;
                %else
                %    fprintf('Deleting %s\n', fullfile(top, directoryNames{i}, ...
                %        folderNames{j}, fileNames{k}));
                %    fprintf(fileID,'%s\n', fullfile(top, directoryNames{i}, ...
                %        folderNames{j}, fileNames{k}));
                %end;
            else
                fprintf('Deleting %s\n', fullfile(top, directoryNames{i}, ...
                        folderNames{j}, fileNames{k}));
                %fprintf(fileID,'%s\n', fullfile(top, directoryNames{i}, ...
                %        folderNames{j}, fileNames{k}));
            end;
        end;
    end;
    if makeVideo,
        close(aviobj);
    end;
end;
fclose(fileID);

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
% and divide by the std dev
dataMean = mean(data(:,:,:,set == 1), 4);
dataStd = std(data(:,:,:,set == 1), 0, 4);
data = bsxfun(@minus, data, dataMean) ;
data = bsxfun(@rdivide, data, dataStd) ;

imdb.images.data = data;
imdb.images.data_mean = dataMean;
imdb.images.std = dataStd;
imdb.images.labels = output;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'};
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),1:5,'uniformoutput',false); 
end