% Author: Rob McCartney
% First visualizes the images being trained
% Runs the declared CNN network 
% Then visualizes the first layer of features
% Uses displayData from Andrew Ng's Machine Learning

addpath code
input('Visualizing a subset of MNIST. Press Enter')
clf ;
data = load(fullfile('data','mnist-baseline', 'imdb.mat')) ;
num = 100 ;
rows = [] ;
for i=1:num,
    rows = [rows ; reshape(data.images.data(:,:,:,i),1,[]) ] ;
end;
displayData(rows) ;

input('Training the classifier. Press Enter')
clf ; 
x = cnn_mnist ;

input('Visualizing the first layer of filters. Press Enter')
clf ;
out = gather(x.layers{1, 1}.filters) ;
rows = [] ;
for i=1:size(out, 4),
    rows = [rows ; reshape(out(:,:,1,i).',1,[]) ] ;
end;
displayData(rows) ;

input('Visualizing the second layer of filters. Press Enter')
clf ;
out = gather(x.layers{1, 3}.filters) ;
rows = [] ;
for i=1:size(out, 4),
    rows = [rows ; reshape(out(:,:,1,i).',1,[]) ] ;
end;
displayData(rows) ;