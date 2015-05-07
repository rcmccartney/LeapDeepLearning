gestures =  {'capE' 'CheckMark' , 'e' , 'F' , 'Figure8' , 'Swipe' , ... 
              'Tap' , 'Grab' , 'Release' , 'Tap2' , 'Wipe' , 'Pinch' };

gestures = cellstr(gestures) ;
testfrac = 0.9;
output = [] ;
images = [] ;

for i=1:length(gestures),
    a = read(VideoReader(strcat(gestures{i}, '.avi'))) ;
    output = [ output  i*ones(1, size(a,4)) ] ;
    for j=1:size(a,4),
        images(:,:,end+1) = rgb2gray(a(:,:,:,j));
    end;
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
data = single(reshape(images,250,150,1,[]));

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